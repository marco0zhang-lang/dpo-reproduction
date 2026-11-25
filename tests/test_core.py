import unittest
import torch
from train import dpo_loss, dataset, TinyLM, logp

class CoreTests(unittest.TestCase):
    def test_initial_loss(self):
        a=torch.tensor([1.,2.]); b=torch.tensor([0.,1.])
        self.assertAlmostEqual(dpo_loss(a,b,a,b,.1).item(), .693147, places=6)
    def test_gradient(self):
        a=torch.tensor([0.],requires_grad=True); b=torch.tensor([0.],requires_grad=True); r=torch.zeros(1)
        dpo_loss(a,b,r,r,.1).backward()
        self.assertLess(a.grad.item(),0); self.assertGreater(b.grad.item(),0)
    def test_split(self):
        train,test=dataset(); self.assertFalse(set(map(tuple,train[0][:,:2].tolist())) & set(map(tuple,test[0][:,:2].tolist())))
    def test_logp(self):
        train,_=dataset(); self.assertEqual(logp(TinyLM(),train[0]).shape,(12,))
if __name__=='__main__': unittest.main()
