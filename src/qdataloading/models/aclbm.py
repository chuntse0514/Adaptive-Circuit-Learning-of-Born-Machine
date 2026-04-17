import pennylane as qml
import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial
from pprint import pprint

from .base import BaseModel
from ..modules.gates import PauliStringRotation
from ..utils.metrics import evaluate
from ..utils.quantum import mutual_information, entanglement_of_formation

def operator_pool(n_qubit, selected_pairs=None):
    pool = []
    gate_description = []
    if selected_pairs:
        for i, j in selected_pairs:
            pool.append(partial(PauliStringRotation, pauli_string='XY', qubit=[i, j]))
            pool.append(partial(PauliStringRotation, pauli_string='XY', qubit=[j, i]))
            pool.append(partial(PauliStringRotation, pauli_string='YZ', qubit=[i, j]))
            pool.append(partial(PauliStringRotation, pauli_string='YZ', qubit=[j, i]))
            pool.append(partial(qml.CRY, wires=[i, j]))
            pool.append(partial(qml.CRY, wires=[j, i]))
            gate_description.extend([f'e^[X{i} Y{j}]', f'e^[X{j} Y{i}]', f'e^[Y{i} Z{j}]', f'e^[Y{j} Z{i}]', f'CRY[{i}, {j}]', f'CRY[{j}, {i}]'])

        for i in range(n_qubit):
            pool.append(partial(qml.RY, wires=i))
            gate_description.append(f'RY[{i}]')
    else:
        for i in range(n_qubit):
            for j in range(n_qubit):
                if i != j:
                    pool.append(partial(PauliStringRotation, pauli_string='XY', qubit=[i, j]))
                    pool.append(partial(PauliStringRotation, pauli_string='YZ', qubit=[i, j]))
                    pool.append(partial(qml.CRY, wires=[i, j]))
                    gate_description.extend([f'e^[X{i} Y{j}]', f'e^[Y{i} Z{j}]', f'CRY[{i}, {j}]'])
        for i in range(n_qubit):
            pool.append(partial(qml.RY, wires=i))
            gate_description.append(f'RY[{i}]')
    return pool, gate_description

def select_qubit_pairs(target_prob, n_qubit, reduction_rate=0.5):
    # target_prob is jnp.ndarray
    state = jnp.sqrt(target_prob)
    ent_list = []
    for i in range(n_qubit):
        for j in range(i+1, n_qubit):
            MI = mutual_information(state, subsystems=(i, j))
            EOF = entanglement_of_formation(state, subsystems=(i, j))
            ent_list.append([(i, j), MI, EOF])
    ent_list = sorted(ent_list, key=lambda x: x[1], reverse=True)
    
    pair_list = [x[0] for x in ent_list]
    MI_list = np.array([x[1] for x in ent_list])
    MI_max = MI_list[1] if len(MI_list) > 1 else MI_list[0]
    N_selected_pairs = np.sum(MI_list > MI_max * reduction_rate)
    return pair_list[:N_selected_pairs]

class ACLBM(BaseModel):
    def __init__(self, data_class, n_epoch: int, n_iter: int, No: int, alpha: float, reduction_rate=None):
        super().__init__(data_class, n_epoch, lr=0.0)
        self.n_iter = n_iter
        self.No = No
        self.alpha = alpha
        self.threshold1 = 1e-3
        self.threshold2 = 5e-3
        
        if reduction_rate is not None:
            selected_pairs = select_qubit_pairs(self.target_prob, self.n_qubit, reduction_rate=reduction_rate)
            self.pool, self.gate_description = operator_pool(self.n_qubit, selected_pairs)
        else:
            self.pool, self.gate_description = operator_pool(self.n_qubit)

        # Initial parameters
        self.params = {
            'Ry-layer': jnp.full((self.n_qubit,), jnp.pi/2),
            'Append': jnp.array([])
        }
        self.operatorID = []
        
        # JAX QNode setup
        self.dev = qml.device('default.qubit', wires=self.n_qubit)
        
        suffix = f", r={reduction_rate}" if reduction_rate is not None else ""
        self.filename = f'{self.image_dir}/ACLBM/ACLBM(data={data_class.name}, No={self.No}, t1={self.threshold1}, t2={self.threshold2}{suffix}).png'
        self.result_file = f'{self.output_dir}/ACLBM/ACLBM(data={data_class.name}, No={self.No}, t1={self.threshold1}, t2={self.threshold2}{suffix}).json'

    def circuit(self, ry_params, append_params, operatorID):
        for q in range(self.n_qubit):
            qml.RY(ry_params[q], wires=q)
        for i, id_ in enumerate(operatorID):
            gate = self.pool[id_]
            gate(append_params[i])
        return qml.probs(wires=range(self.n_qubit))

    def eval_circuit(self, ry_params, append_params, operatorID, eval_params):
        # Basis circuit
        for q in range(self.n_qubit):
            qml.RY(ry_params[q], wires=q)
        for i, id_ in enumerate(operatorID):
            gate = self.pool[id_]
            gate(append_params[i])
        # Evaluation layer (one parameter for each operator in pool)
        for i, gate in enumerate(self.pool):
            gate(eval_params[i])
        return qml.probs(wires=range(self.n_qubit))

    def select_operator(self, randomize=False):
        qnode = qml.QNode(self.eval_circuit, self.dev, interface='jax')
        
        def loss_fn(eval_p):
            prob = qnode(self.params['Ry-layer'], self.params['Append'], self.operatorID, eval_p)
            mask = (self.target_prob > 0) & (prob > 0)
            return jnp.sum(self.target_prob[mask] * jnp.log(self.target_prob[mask] / prob[mask]))

        grads = jax.grad(loss_fn)(jnp.zeros(len(self.pool)))
        grads_abs = jnp.abs(grads)
        
        selected_index = jnp.argsort(grads_abs)[::-1][:self.No]
        selected_index = np.array(selected_index).tolist()
        selected_gate = [self.gate_description[index] for index in selected_index]
        max_grad = [float(grads_abs[index]) for index in selected_index]
        
        if randomize:
            perm_index = np.random.permutation(len(max_grad))
            max_grad = [max_grad[i] for i in perm_index]
            selected_index = [selected_index[i] for i in perm_index]
            selected_gate = [selected_gate[i] for i in perm_index]

        return max_grad, selected_index, selected_gate

    def fit(self):
        qnode = qml.QNode(self.circuit, self.dev, interface='jax')
        
        def criterion(p, q):
            mask = (p > 0) & (q > 0)
            return jnp.sum(p[mask] * jnp.log(p[mask] / q[mask]))

        for i_iter in range(self.n_iter):
            max_grad, selected_index, selected_gate = self.select_operator(randomize=True)
            pprint(f'==== Found maximium gradient {max_grad} of gate ' + ', '.join(selected_gate) + ' ====')
            
            if np.max(max_grad) < self.threshold1:
                print('Convergence criterion reached.')
                break

            self.operatorID.extend(selected_index)
            self.params['Append'] = jnp.concatenate([self.params['Append'], jnp.zeros(len(max_grad))])
            
            lr = float(jnp.linalg.norm(jnp.array(max_grad))) / np.sqrt(self.No) * self.alpha
            print('learning rate = ', lr)
            
            optimizer = optax.adam(lr, b1=0.7)
            opt_state = optimizer.init(self.params)

            @jax.jit
            def train_step(params, opt_state, operatorID):
                def loss_fn(p):
                    prob = qnode(p['Ry-layer'], p['Append'], operatorID)
                    return criterion(self.target_prob, prob), prob
                (loss, prob), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                grad_norm = jnp.linalg.norm(jnp.concatenate([grads['Ry-layer'], grads['Append']]))
                return params, opt_state, loss, prob, grad_norm

            while True:
                self.params, opt_state, loss, prob, grad_norm = train_step(self.params, opt_state, tuple(self.operatorID))
                
                self.loss_history.append(float(loss))
                kl_div, js_div = evaluate(self.target_prob, prob)
                self.kl_history.append(kl_div)
                self.js_history.append(js_div)
                self.grad_norm_history.append(float(grad_norm))

                if grad_norm < self.threshold2:
                    break
                if len(self.loss_history) >= self.n_epoch:
                    break

            print(f"iteration: {i_iter+1} | total_epochs: {len(self.loss_history)} | loss: {float(loss):.6f} | KL: {kl_div:.6f}")
            if len(self.loss_history) >= self.n_epoch:
                break
            
        self.plot_training_result(prob, self.filename)
        self.save_results(prob, self.result_file)
