import jax
import jax.numpy as jnp
import optax
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from .base import BaseModel
from ..modules.ansatz import get_generator1, get_generator2
from ..modules.losses import make_K, mmd_loss
from ..utils.metrics import evaluate

def mutual_information_classical(pdata):
    pdata = np.array(pdata)
    sl = [0, 1]  # possible states
    d = len(sl)  # number of possible states
    num_bit = int(np.round(np.log(len(pdata))/np.log(2)))
    basis = np.arange(2**num_bit, dtype='uint32')

    pxy = np.zeros([num_bit, num_bit, d, d])
    px = np.zeros([num_bit, d])
    pdata2d = np.broadcast_to(pdata[:,None], (len(pdata), num_bit))
    pdata3d = np.broadcast_to(pdata[:,None,None], (len(pdata), num_bit, num_bit))
    offsets = np.arange(num_bit-1,-1,-1)

    for s_i in sl:
        mask_i = (basis[:,None]>>offsets)&1 == s_i
        px[:,s_i] = np.ma.array(pdata2d, mask=~mask_i).sum(axis=0)
        for s_j in sl:
            mask_j = (basis[:,None]>>offsets)&1 == s_j
            pxy[:,:,s_i,s_j] = np.ma.array(pdata3d, mask=~(mask_i[:,None,:]&mask_j[:,:,None])).sum(axis=0)

    # mutual information
    pratio = pxy/np.maximum(px[:,None,:,None]*px[None,:,None,:], 1e-15)
    for i in range(num_bit):
        pratio[i, i] = 1
    I = (pxy*np.log(pratio)).sum(axis=(2,3))
    return I

def chowliu_tree(pdata):
    X = mutual_information_classical(pdata)
    Tcsr = -minimum_spanning_tree(-X)
    Tcoo = Tcsr.tocoo()
    pairs = list(zip(Tcoo.row, Tcoo.col))
    print(f'Chow-Liu tree pairs = {pairs}')
    return pairs

class QCBM(BaseModel):
    def __init__(self, data_class, n_epoch: int, reps: int, lr: float):
        super().__init__(data_class, n_epoch, lr)
        self.reps = reps
        
        # Init parameters and QNode
        if data_class.dist_property == 'sparse':
            pairs = chowliu_tree(self.target_prob)
            self.qnode, param_shape = get_generator1(self.n_qubit, reps, pairs)
            self.log_mmd = False
        else:
            pairs = chowliu_tree(self.target_prob)
            self.qnode, param_shape = get_generator2(self.n_qubit, reps, pairs)
            self.log_mmd = True

        # JAX parameter initialization
        key = jax.random.PRNGKey(42)
        self.params = jax.random.normal(key, param_shape) * (jnp.pi / 8)
        
        self.K = make_K(self.n_qubit, [0.5, 1., 2., 4.])
        
        # Optax optimizer
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params)

        self.filename = f'{self.image_dir}/QCBM/QCBM(data={data_class.name}, lr={lr}, reps={reps}).pdf'
        self.result_file = f'{self.output_dir}/QCBM/QCBM(data={data_class.name}, lr={lr}, reps={reps}).json'

    def fit(self, save=True):
        threshold = 1e-5 if self.data_class.dist_property == 'sparse' else 1e-3
        
        @jax.jit
        def step(params, opt_state):
            def loss_fn(p):
                pred_prob = self.qnode(p, self.n_qubit, self.reps)
                return mmd_loss(self.target_prob, pred_prob, self.K, log=self.log_mmd), pred_prob
            
            (loss, pred_prob), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, pred_prob, grads

        params = self.params
        opt_state = self.opt_state

        for i_epoch in range(self.n_epoch):
            params, opt_state, loss, pred_prob, grads = step(params, opt_state)
            
            grad_norm = jnp.linalg.norm(grads)
            
            self.loss_history.append(float(loss))
            kl_div, js_div = evaluate(self.target_prob, pred_prob)
            self.kl_history.append(kl_div)
            self.js_history.append(js_div)
            self.grad_norm_history.append(float(grad_norm))

            if (i_epoch + 1) % 100 == 0:
                loss_type = "MMD" if not self.log_mmd else "Log MMD"
                print(f'epoch: {i_epoch+1} | {loss_type} loss: {float(loss):6f} | KL divergence: {kl_div:6f} | JS divergence: {js_div:6f} | Grad Norm: {float(grad_norm):6f}')

            if grad_norm < threshold:
                print(f"Converged at epoch {i_epoch+1}")
                break
        
        if save:
            self.plot_training_result(pred_prob, self.filename)
            self.save_results(pred_prob, self.result_file)
        self.params = params
