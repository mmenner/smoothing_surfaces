import numpy as np
from scipy import sparse
from scipy.linalg import block_diag
from gurobipy import Model, QuadExpr, GRB, quicksum
from numpy import empty
from cvxopt import matrix
from cvxopt import solvers as cvxsolvers
from scipy import stats
import sys 

tt = np.matrix.transpose
cc = np.concatenate

class QPSmoother:
    '''QPSmoother class. 
    init:
          x1:      full grid dimension1  
          x1_obs:  features dimension1 (must intersect with full grid x1)
          x2:      full grid dimension2
          x2_obs:  features dumension2 (must intersect with full grid x2)
          y_obs:   observed label
          l_diff:  smoothing parameter
          hessian: hessian in the QP problem. Computed if not provided
    get_qp:
        sets up matrices for the qp solver and saves them in QPSmoother.qp
    solve: 
        solve qp problem using a given solver
    '''
    def __init__(self, x1=None, x1_obs=None, x2=None, x2_obs=None, y_obs=None, l_diff=1, hessian=None):
        
        if min(np.diff(x1))<0 or len(np.unique(x1)) < len(x1):
            print('Error: x1 must be strictly increasing')
            sys.exit()   
        self.hessian = hessian
        if x2_obs is not None:
            self.dimension = 2
            if min(np.diff(x2))<0 or len(np.unique(x2)) < len(x2):
                print('Error: x2 must be strictly increasing')
                sys.exit()    
            if len(np.unique(list(zip(x1_obs,x2_obs)))) < len(x2_obs) or len(x2_obs) != len(x2_obs):
                print('Error: x1_obs, x2_obs must consits of unique pairs and have both the same length as y_obs')
                sys.exit() 
            # compute 2d - hessian and store
            if self.hessian is None:
                self.hessian = hessian2d(x1, x2, l_diff)
            self.l_diff = l_diff
        else:
            self.dimension = 1              
            if len(np.unique(x1_obs)) < len(x1_obs):
                print('Error: values of x1_obs must be unique')
                sys.exit()
            if len(x1_obs) != len(y_obs):
                print('Error: x1_obs must have the same length as y_obs')
                sys.exit()                
            #sort values (x1_obs must be increasing)
            idx = np.argsort(x1_obs) 
            x1_obs = x1_obs[idx]
            y_obs = y_obs[idx]
            if self.hessian is None:
                self.hessian = hessian1d(len(x1), 1, -2, 1)
        self.x1 = x1
        self.x1_obs = x1_obs
        self.x2 = x2
        self.x2_obs = x2_obs
        self.y_obs = y_obs
        self.qp = None
        self.y = None       
        
    def get_qp(self, qp_method, solver, l_fit=0, lb=-np.infty, ub=np.infty):
        
        if qp_method == 'smooth':
            if l_fit <= 0 or l_fit >=1:
                print('l_fit must lie in the interval (0,1). l_fit is set to nearest boundary')
            l_fit = min(max(1e-10,l_fit),1-1e-10)
            l_fit = (stats.norm.ppf(l_fit)+2.5)/5
            l_fit = max(l_fit,0)**4
            if self.dimension == 1:   
                l_fit = l_fit*len(self.x1)**2/len(self.y_obs)/((self.x1[-1]-self.x1[0]))
            else:
                l_fit = l_fit*(len(self.x1)*len(self.x2))**2/len(self.y_obs)/((self.x1[-1]-self.x1[0])*(self.x2[-1]-self.x2[0]))        
        self.qp = QPComponents(self, qp_method, l_fit)
        self.qp.method = qp_method
        self.qp.l_fit_init = l_fit
        if qp_method == 'smooth':
            l_fit = (stats.norm.ppf(min(max(1e-10,l_fit),1-1e-10))+2.5)/5
            l_fit = max(l_fit,0)**4
            if self.dimension == 1:
                l_fit = l_fit*len(self.x1)**2/len(self.y_obs)/((self.x1[-1]-self.x1[0]))
            else:
                l_fit = l_fit*(len(self.x1)*len(self.x2))**2/len(self.y_obs)/((self.x1[-1]-self.x1[0])*(self.x2[-1]-self.x2[0]))
        self.qp.l_fit = l_fit
        self.qp.ub = ub
        self.qp.lb = lb

              
    def solve(self, solver, try_closed_form = False):
        if solver not in ['gurobi', 'cvxopt', 'closed_form']:
            print('solver needs to be either gurobi, cvxopt, or closed_form')
            sys.exit()
        if self.qp.method == 'exact' and solver == 'cvxopt':
            print('CVXOPT does currently not work with the method exact. Use gurobi or closed_form instead')
            sys.exit()
        if self.qp.lb == -np.infty and self.qp.ub == np.infty and solver != 'closed_form':
            print('No lower and upper bounds set, change solver to closed_form')
            solver = 'closed_form'
        if (self.qp.lb != -np.infty or self.qp.ub != np.infty) and solver == 'closed_form':
            print('WARNING: lower and upper bounds set but solver is closed_form. Solution might be out of range')          
        if try_closed_form is True and (self.qp.lb != -np.infty or self.qp.ub != np.infty):
            qp_solver = QPSolver('closed_form')
            qp_solver.optimize(self.qp)
            if min(qp_solver.solution) < self.qp.lb or max(qp_solver.solution) > self.qp.ub:
                try_closed_form = False
                if solver == 'closed_form':
                    try_closed_form = True
            else:
                print('Optimal solution found with closed_form')
        if try_closed_form is False:
            if solver == 'gurobi' and self.qp.method == 'exact' and (self.qp.l_fit > 1 or self.qp.l_fit < 0):
                print('Gurobi requires l_fit to lie between 0 and 1. Set l_fit to next bound')
                self.qp.l_fit = min(max(self.qp.l_fit,0),1)   
            qp_solver = QPSolver(solver)
            qp_solver.optimize(self.qp)
        self.y = qp_solver.solution
        self.y_obs_solution = self.y[self.qp.idx]
        if self.dimension == 2:
            self.y_vec = self.y  
            self.y = np.array(self.y.reshape((self.qp.n1, self.qp.n2)))
            self.x1_mesh, self.x2_mesh = np.meshgrid(self.x1, self.x2)  
        self.qp.solver = solver
        return self.y


def hessian1d(n, d, m, u):
    # creates a hessian-type matrix
    h1d = sparse.identity(n).toarray() * m
    idc_0 = np.array(range(1, n))
    idc_1 = np.array(range(0, n - 1))
    h1d[(idc_0, idc_1)] = d
    h1d[(idc_1, idc_0)] = u
    h1d[0][0] += d
    h1d[-1][-1] += u
    return h1d


def hessian2d(x, y, l_diff):
    #create a hessian-type block matrix to smooth surfaces
    #can be extended to non-equidistant steps within x or y
    n_x = len(x)
    n_y = len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dxy = l_diff / (4 * dx * dy)
    dyy = 1 / (dy ** 2)
    dxx = 1 / (dx ** 2)
    m = hessian1d(n_x, dxx, -2 * (dxx + dyy), dxx)
    u = hessian1d(n_x, -dxy, dyy, dxy)
    d = hessian1d(n_x, dxy, dyy, -dxy)
    m_blk = block_diag(m, m)
    u_blk = u
    d_blk = d
    # set together
    for iy in range(0, n_y - 4):
        m_blk = block_diag(m_blk, m)
        u_blk = block_diag(u_blk, u)
        d_blk = block_diag(d_blk, d)
    zero1 = sparse.csr_matrix((n_x * (n_y - 2), n_x)).todense()
    zero2 = sparse.csr_matrix((n_x * (n_y - 3), n_y)).todense()
    u_blk_filled = cc((zero1, cc((u_blk, tt(zero2)))), axis=1)
    d_blk_filled = cc((tt(zero1), cc((d_blk, zero2), axis=1)))
    cd_h_center = m_blk + u_blk_filled + d_blk_filled
    cd_h_top = cc((cc((m + d, u), axis=1), tt(zero1)), axis=1)
    cd_h_middle = cc((cc((cc((d, zero2)), cd_h_center), axis=1), cc((zero2, u))), axis=1)
    cd_h_down = cc((tt(zero1), cc((d, m + u), axis=1)), axis=1)
    h2d = cc((cd_h_top, cc((cd_h_middle, cd_h_down))))
    #possible scaling can be implemented here (function of x and y)
    #h2d = tt(cdh).dot(cdh)
    return np.array(h2d)


class QPComponents:
    def __init__(self, qp_input, method, l_fit):
        self.H = None
        self.f = None
        self.A = None
        self.b = None
        self.Aeq = None
        self.beq = None
        self.n1 = len(qp_input.x1) 
        if qp_input.x2 is not None:
            self.n2 = len(qp_input.x2)
            self.n = self.n1 * self.n2
        else:
            self.n = self.n1
            self.n2 = None
        self.n_obs = len(qp_input.y_obs)
        self.qp_generate(qp_input, method, l_fit)

    def qp_generate(self, qp_input, method, l_fit):
        self.H = tt(qp_input.hessian).dot(qp_input.hessian)/self.n
        self.f = np.zeros(self.n)
        if qp_input.dimension == 1:
            self.idx = np.where(np.in1d(qp_input.x1, qp_input.x1_obs))[0]
        elif qp_input.dimension == 2:
            idx1 = []
            for x1_obs_i in qp_input.x1_obs:
                  idx1.append(np.where(x1_obs_i == qp_input.x1)[0][0])
                  idx2 = []
            for x2_obs_i in qp_input.x2_obs:
                 idx2.append(np.where(x2_obs_i == qp_input.x2)[0][0])
            self.idx = self.n1 * np.array(idx1) + np.array(idx2)
            self.idx1 = idx1
            self.idx2 = idx2
        if method == 'exact':
             self.H = self.H - l_fit*qp_input.hessian
             self.Aeq = np.array(sparse.csr_matrix((self.n_obs, self.n)).todense())
             for iterate, i in enumerate(self.idx):
                  self.Aeq[iterate][i] = 1
             self.beq = qp_input.y_obs      
        elif method == 'smooth':
             for iterate, i in enumerate(self.idx):
                  self.H[i][i] = self.H[i][i] + l_fit / self.n_obs
                  self.f[i] = -l_fit * qp_input.y_obs[iterate] / self.n_obs
        
          
def get_nonzero_rows(M):
    nonzero_rows = {}
    rows, cols = M.nonzero()
    for ij in zip(rows, cols):
        i, j = ij
        if i not in nonzero_rows:
            nonzero_rows[i] = []
        nonzero_rows[i].append(j)
    return nonzero_rows


class QPSolver:
    def __init__(self, solver_name):
        self.solver_name = solver_name
        self.solution = None

    def optimize(self, qp):
        if self.solver_name == 'gurobi':
            self.gurobi_qp(qp)
        elif self.solver_name == 'cvxopt':
            self.cvx_qp(qp)
        elif self.solver_name == 'closed_form':
            self.closed_form(qp)

    def closed_form(self, qp):
        if qp.H is not None:
            if qp.Aeq is not None:
                A_lagrange1 = cc((qp.H, tt(qp.Aeq)), axis=1)
                A_lagrange2 = cc((qp.Aeq, np.zeros((len(qp.beq), len(qp.beq)))), axis=1)
                A_lagrange = cc((A_lagrange1, A_lagrange2))
                b_lagrange = cc((-qp.f, qp.beq))
                solution_lagrange = np.linalg.solve(A_lagrange, b_lagrange)
                self.solution = solution_lagrange[0:len(qp.H)]
            else:
                self.solution = np.linalg.solve(qp.H, -qp.f)
        else:
            self.solution = np.linalg.solve(qp.Aeq, qp.beq)

    def cvx_qp(self,qp):
        A_cvx = cc((-sparse.identity(qp.n).toarray(), sparse.identity(qp.n).toarray()))
        b_cvx = cc((np.array(-qp.lb)*np.ones(qp.n), np.array(qp.ub)*np.ones(qp.n)))
        if qp.A is not None:
             A_cvx = matrix(cc((qp.A, A_cvx)))
             b_cvx = matrix(cc((qp.b,b_cvx)))
        if qp.Aeq is not None:
             cvxopt_dict = cvxsolvers.qp(matrix(qp.H), matrix(qp.f), matrix(A_cvx), matrix(b_cvx), matrix(qp.Aeq), matrix(qp.beq))
        else:
             cvxopt_dict = cvxsolvers.qp(matrix(qp.H), matrix(qp.f), matrix(A_cvx), matrix(b_cvx))
        self.solution = np.array(list(cvxopt_dict['x'])) 
       
    def gurobi_qp(self, qp):
        n = qp.H.shape[1]
        model = Model()
        x = {
            i: model.addVar(
                vtype=GRB.CONTINUOUS,
                name='x_%d' % i,
                lb=qp.lb,
                ub=qp.ub)
            for i in range(n)
        }
        model.update()  
        obj = QuadExpr()
        rows, cols = qp.H.nonzero()
        for i, j in zip(rows, cols):
            obj += 0.5 * x[i] * qp.H[i, j] * x[j]
        for i in range(n):
            obj += qp.f[i] * x[i]
        model.setObjective(obj, GRB.MINIMIZE)
        if qp.A is not None:
            A_nonzero_rows = get_nonzero_rows(qp.A)
            for i, row in A_nonzero_rows.items():
                model.addConstr(quicksum(qp.A[i, j] * x[j] for j in row) <= qp.b[i])

        if qp.Aeq is not None:
            A_nonzero_rows = get_nonzero_rows(qp.Aeq)
            for i, row in A_nonzero_rows.items():
                model.addConstr(quicksum(qp.Aeq[i, j] * x[j] for j in row) == qp.beq[i])
        model.optimize()
        self.solution = empty(n)
        for i in range(n):
            self.solution[i] = model.getVarByName('x_%d' % i).x
