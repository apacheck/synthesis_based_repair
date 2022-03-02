import torch
import numpy as np
import time

EPS = 1e-8


class TermStatic:
    def __init__(self, x):
        self.x = x

    def eval(self, t):
        return self.x


class TermDynamic:
    def __init__(self, xs):
        assert xs.dim() == 3, "Dynamic term needs 3 dims: Batch_Size x Time_Steps x Spatial_Dims"
        self.xs = xs

    def eval(self, t):
        return self.xs[:, t]


class BoolConst:
    def __init__(self, x):
        self.x = x.float()  ## True is 1, False is 0

    def loss(self, t):
        return 1.0 - self.x

    def satisfy(self, t):
        return self.x > (1 - EPS)


class EQ:
    """ a == b"""

    def __init__(self, term_a, term_b):
        self.term_a = term_a
        self.term_b = term_b

    def loss(self, t):
        a = self.term_a.eval(t)
        b = self.term_b.eval(t)

        return torch.norm(a - b, dim=1)

    def satisfy(self, t):
        return (self.term_a.eval(t) == self.term_b.eval(t)).all(1)


class GEQ:
    """ a >= b """

    def __init__(self, term_a, term_b):
        self.term_a = term_a
        self.term_b = term_b

    def loss(self, t):
        a = self.term_a.eval(t)
        b = self.term_b.eval(t)
        return (b - a).clamp(min=0.0).sum(1)

    def satisfy(self, t):
        return (self.term_a.eval(t) >= self.term_b.eval(t)).all(1)


class GEQ2:
    """ a >= b """

    def __init__(self, term_a, term_b, dim=[0, 1], multiplier=1):
        self.term_a = term_a
        self.term_b = term_b
        self.dim = dim
        self.mutliplier = multiplier
        assert term_b.x.shape[0] == len(dim)

    def loss(self, t):
        a = self.term_a.eval(t)[:, self.dim]
        b = self.term_b.eval(t)
        return (b - a).clamp(min=0.0).sum(1)
        # return torch.square((b - a).clamp(min=0.0)).sum(1)

    def satisfy(self, t):
        return (self.term_a.eval(t)[:, self.dim] >= self.term_b.eval(t)).all(1)


class LEQ2:
    """ a >= b """

    def __init__(self, term_a, term_b, dim=[0, 1], multiplier=1):
        self.term_a = term_a
        self.term_b = term_b
        self.dim = dim
        self.mutliplier = multiplier
        assert term_b.x.shape[0] == len(dim)

    def loss(self, t):
        a = self.term_a.eval(t)[:, self.dim]
        b = self.term_b.eval(t)
        return (a - b).clamp(min=0.0).sum(1)
        # return torch.square((a - b).clamp(min=0.0)).sum(1)

    def satisfy(self, t):
        return (self.term_b.eval(t) >= self.term_a.eval(t)[:, self.dim]).all(1)


class LEQ:
    """ a <= b """

    def __init__(self, term_a, term_b):
        self.term_a = term_a
        self.term_b = term_b

    def loss(self, t):
        a = self.term_a.eval(t)
        b = self.term_b.eval(t)
        return (a - b).clamp(min=0.0).sum(1)

    def satisfy(self, t):
        return (self.term_a.eval(t) <= self.term_b.eval(t)).all(1)


class GT:
    """ a > b """

    def __init__(self, term_a, term_b):
        self.term_a = term_a
        self.term_b = term_b

    def loss(self, t):
        a = self.term_a.eval(t)
        b = self.term_b.eval(t)
        equality = (a == b).all(1).type(a.type())  # strict greater than, so equality penalized
        return (b - a).clamp(min=0.0).sum(1) + equality

    def satisfy(self, t):
        return (self.term_a.eval(t) > self.term_b.eval(t)).all(1)


class GT2:
    """ a > b """

    def __init__(self, term_a, term_b, dim=[0, 1], multiplier=1):
        self.term_a = term_a
        self.term_b = term_b
        self.dim = dim
        self.mutliplier = multiplier
        assert term_b.x.shape[0] == len(dim)

    def loss(self, t):
        a = self.term_a.eval(t)[:, self.dim]
        b = self.term_b.eval(t)
        equality = (a == b).all(1).type(a.type())  # strict greater than, so equality penalized
        return (b - a).clamp(min=0.0).sum(1) + equality
        # return torch.square((b - a).clamp(min=0.0)).sum(1) + equality

    def satisfy(self, t):
        return (self.term_a.eval(t)[:, self.dim] > self.term_b.eval(t)).all(1)


class LT:
    """ a < b """

    def __init__(self, term_a, term_b):
        self.term_a = term_a
        self.term_b = term_b

    def loss(self, t):
        a = self.term_a.eval(t)
        b = self.term_b.eval(t)
        equality = (a == b).all(1).type(a.type())  # strict greater than, so equality penalized
        return (a - b).clamp(min=0.0).sum(1) + equality

    def satisfy(self, t):
        return (self.term_a.eval(t) < self.term_b.eval(t)).all(1)


class LT2:
    """ a < b """

    def __init__(self, term_a, term_b, dim=[0, 1], multiplier=1):
        self.term_a = term_a
        self.term_b = term_b
        self.dim = dim
        self.mutliplier = multiplier
        assert term_b.x.shape[0] == len(dim)

    def loss(self, t):
        a = self.term_a.eval(t)[:, self.dim]
        b = self.term_b.eval(t)
        equality = (a == b).all(1).type(a.type())  # strict greater than, so equality penalized
        return (a - b).clamp(min=0.0).sum(1) + equality

    def satisfy(self, t):
        return (self.term_a.eval(t)[:, self.dim] < self.term_b.eval(t)).all(1)


class And:
    """ E_1 and E_2 and ... E_k"""

    def __init__(self, exprs):
        self.exprs = exprs

    def loss(self, t):
        losses = torch.stack([exp.loss(t) for exp in self.exprs])
        return soft_maximum(losses, 0)
        # return torch.sum(losses, dim=0, keepdim=True)

    def satisfy(self, t):
        sats = torch.stack([exp.satisfy(t) for exp in self.exprs])
        return sats.all(0, keepdim=False)


class Or:
    """ E_1 or E_2 or .... E_k"""

    def __init__(self, exprs):
        self.exprs = exprs

    def loss(self, t):
        losses = torch.stack([exp.loss(t) for exp in self.exprs])
        return soft_minimum(losses, 0)
        # return torch.prod(losses, dim=0, keepdim=True)

    def satisfy(self, t):
        sats_raw = [exp.satisfy(t) for exp in self.exprs]
        sats = torch.stack(sats_raw)
        return sats.any(0, keepdim=False)


class Implication:
    """ A -> B """

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.imp = Or([Negate(a), b])

    def loss(self, t):
        return self.imp.loss(t)

    def satisfy(self, t):
        return self.imp.satisfy(t)


class Next:
    """ N(X) """

    def __init__(self, exp):
        self.exp = exp

    def loss(self, t):
        return self.exp.loss(t + 1)

    def satisfy(self, t):
        return self.exp.satisfy(t + 1)


class Always:
    """ Always X """

    def __init__(self, exp, max_t):
        self.exp = exp
        self.max_t = max_t

    def loss(self, t):
        assert t <= self.max_t
        losses = torch.stack([self.exp.loss(i) for i in range(t, self.max_t)], 1)
        # return torch.sum(losses, dim=1)
        return soft_maximum(losses, 1)

    def satisfy(self, t):
        assert t <= self.max_t
        sats = torch.stack([self.exp.satisfy(i) for i in range(t, self.max_t)], 1)
        return sats.all(1)


class Eventually:
    """ Eventually X """

    def __init__(self, exp, max_t):
        self.exp = exp
        self.max_t = max_t

    def loss(self, t):
        assert t <= self.max_t
        losses = torch.stack([self.exp.loss(i) for i in range(t, self.max_t)], 1)
        return soft_minimum(losses, 1)

    def satisfy(self, t):
        assert t <= self.max_t
        sats = torch.stack([self.exp.satisfy(i) for i in range(t, self.max_t)], 1)
        return sats.any(1)


def soft_maximum(xs, dim, p=5000):
    ln_N = np.log(xs.shape[dim])
    return ((xs * p).logsumexp(dim) - ln_N) / p


def soft_minimum(xs, dim, p=5000):
    ln_N = np.log(xs.shape[dim])
    return ((xs * -p).logsumexp(dim) - ln_N) / (-p)


class Until1:
    def __init__(self, a, b, max_t):
        self.a = a
        self.b = b
        self.max_t = max_t

    def loss(self, t):
        # raise NotImplementedError()
        assert t <= self.max_t

        # New
        # t_start = time.time()
        # # b_losses = torch.stack([self.b.loss(jj) for jj in range(t, self.max_t)])
        # sum_exp_b_losses = torch.cumsum(torch.exp(torch.stack([self.b.loss(jj) for jj in range(t, self.max_t)]) * -200), dim=0)
        # # sum_exp_b_losses = torch.cumsum(exp_b_losses, dim=0)
        # exp_a_losses = torch.exp(torch.stack([self.a.loss(jj) for jj in range(t, self.max_t)]) * -200)
        # # exp_a_losses = torch.exp(a_losses * -200)
        # a_plus_b_losses_original = (torch.log(exp_a_losses + sum_exp_b_losses) - torch.transpose(torch.log(torch.arange(t+1, self.max_t+1)).repeat(32, 1), 1, 0)) / (-200)
        # t_mid = time.time()

        # Old 2
        # b_losses = torch.stack([self.b.loss(jj) for jj in range(t, self.max_t)])
        # exp_b_losses = torch.exp(b_losses * -200)
        # sum_exp_b_losses = torch.cumsum(exp_b_losses, dim=0)
        # a_losses = torch.stack([self.a.loss(jj) for jj in range(t, self.max_t)])
        # exp_a_losses = torch.exp(a_losses * -200)
        # a_plus_b_losses = (torch.log(exp_a_losses + sum_exp_b_losses) - torch.transpose(torch.log(torch.arange(t+1, self.max_t+1)).repeat(32, 1), 1, 0)) / (-200)

        # Old
        b_losses = torch.stack([self.b.loss(jj) for jj in range(t, self.max_t)])
        losses = soft_minimum(torch.stack([self.a.loss(t), self.b.loss(t)]), 0)[None, :]
        for ii in range(t + 1, self.max_t):
            one_loss = soft_minimum(torch.cat([self.a.loss(ii)[None, :], b_losses[:(ii + 1), :]]), 0)
            losses = torch.cat([losses, one_loss[None, :]], 0)
        #
        # print("New: {} Old: {}".format(t_mid-t_start, time.time()-t_mid))
        return soft_maximum(losses, 0)

    def satisfy(self, t):
        sats_a = torch.stack([self.a.satisfy(i) for i in range(t, self.max_t)])
        sats_b = torch.stack([self.b.satisfy(i) for i in range(t, self.max_t)])

        eventually_b = sats_b.any(dim=0, keepdim=False)
        bs_onward = torch.cumsum(sats_b.int(), dim=0) > 0

        keep_a_until = (sats_a | bs_onward).all(dim=0, keepdim=False)

        return eventually_b & keep_a_until


class Until2:
    def __init__(self, a, b, max_t):
        self.a = a
        self.b = b
        self.max_t = max_t

    def loss(self, t):
        # raise NotImplementedError()
        assert t <= self.max_t
        losses = soft_minimum(torch.stack([self.a.loss(t), self.b.loss(t)]), 0)[None, :]
        b_losses = torch.stack([self.b.loss(jj) for jj in range(t, self.max_t)])
        for ii in range(t + 1, self.max_t):
            one_loss = soft_minimum(torch.cat([self.a.loss(ii)[None, :], b_losses[:(ii + 1), :]]), 0)
            losses = torch.cat([losses, one_loss[None, :]], 0)

        return soft_maximum(losses, 0)

    def satisfy(self, t):
        sats_a = torch.stack([self.a.satisfy(i) for i in range(t, self.max_t)])
        sats_b = torch.stack([self.b.satisfy(i) for i in range(t, self.max_t)])

        eventually_b = sats_b
        bs_onward = torch.cumsum(sats_b.int(), dim=0) > 0

        keep_a_until = (sats_a | bs_onward)

        return eventually_b & keep_a_until


class Negate:
    """ !X """

    def __init__(self, exp):
        self.exp = exp

        if isinstance(self.exp, LT):
            self.neg = GEQ(self.exp.term_a, self.exp.term_b)
        elif isinstance(self.exp, GT):
            self.neg = LEQ(self.exp.term_a, self.exp.term_b)
        elif isinstance(self.exp, EQ):
            self.neg = Or([LT(self.exp.term_a, self.exp.term_b), LT(self.exp.term_b, self.exp.term_a)])
        elif isinstance(self.exp, LEQ):
            self.neg = GT(self.exp.term_a, self.exp.term_b)
        elif isinstance(self.exp, GEQ):
            self.neg = LT(self.exp.term_a, self.exp.term_b)

        if isinstance(self.exp, LT2):
            # geq_list = []
            # for dim in self.exp.dim:
            #     geq_list.append(GEQ2(self.exp.term_a[dim], self.exp.term_b[dim], dim))
            # self.neg = Or(geq_list)
            self.neg = GEQ2(self.exp.term_a, self.exp.term_b, self.exp.dim)
        elif isinstance(self.exp, GT2):
            # leq_list = []
            # for dim in self.exp.dim:
            #     leq_list.append(LEQ2(self.exp.term_a[dim], self.exp.term_b[dim], dim))
            # self.neg = Or(leq_list)
            self.neg = LEQ2(self.exp.term_a, self.exp.term_b, self.exp.dim)
        elif isinstance(self.exp, LEQ2):
            # g_list = []
            # for dim in self.exp.dim:
            #     g_list.append(GT2(self.exp.term_a[dim], self.exp.term_b[dim], dim))
            # self.neg = Or(g_list)
            self.neg = GT2(self.exp.term_a, self.exp.term_b, self.exp.dim)
        elif isinstance(self.exp, GEQ2):
            # l_list = []
            # for dim in self.exp.dim:
            #     l_list.append(LT2(self.exp.term_a[dim], self.exp.term_b[dim], dim))
            # self.neg = Or(l_list)
            self.neg = LT2(self.exp.term_a, self.exp.term_b, self.exp.dim)

        elif isinstance(self.exp, And):
            neg_exprs = [Negate(e) for e in self.exp.exprs]
            self.neg = Or(neg_exprs)
        elif isinstance(self.exp, Or):
            neg_exprs = [Negate(e) for e in self.exp.exprs]
            self.neg = And(neg_exprs)
        elif isinstance(self.exp, Implication):
            self.neg = And([self.exp.a, Negate(self.exp.b)])
        elif isinstance(self.exp, BoolConst):
            self.neg = BoolConst(1.0 - self.exp.x)
        elif isinstance(self.exp, Next):
            self.neg = Next(Negate(self.exp.exp))
        elif isinstance(self.exp, Eventually):
            self.neg = Always(Negate(self.exp.exp), self.exp.max_t)
        elif isinstance(self.exp, Always):
            self.neg = Eventually(Negate(self.exp.exp), self.exp.max_t)
        elif isinstance(self.exp, Until1):
            part_a = Until1(And([self.exp.a, Negate(self.exp.b)]),
                           And([Negate(self.exp.a), Negate(self.exp.b)]), self.exp.max_t)
            part_b = Always(And([self.exp.a, Negate(self.exp.b)]), self.exp.max_t)
            self.neg = Or([part_a, part_b])
        # elif isinstance(self.exp, Negate):
        #     self.neg = Negate(self.exp.exp)
        else:
            assert False, 'Class not supported %s' % str(type(exp))

    def loss(self, t):
        return self.neg.loss(t)

    def satisfy(self, t):
        return self.neg.satisfy(t)
