#Uses python3

import numpy as np
import pandas as pd 
import time
import sys
import cy

#====================== all functions ======================
# neighbours  =============================================
def coord(site):
    """get coordinate i of vector"""
    x = site // L
    y = site - x*L
    return (x,y)

def get(i):
    """fixin' boundary"""
    if i<0: return i
    else: return i % L
    
def get_neigh():
    """get neighbour's arr"""
    s = np.arange(L**2).reshape(L,L)
    nei = []
    for site in range(L*L):
        i,j = coord(site)
        nei += [s[get(i-1),get(j-1)],s[get(i-1),get(j)],s[get(i),get(j+1)],
                s[get(i+1),get(j+1)],s[get(i+1),get(j)],s[get(i),get(j-1)]]
    return np.array(nei, dtype=np.int32).reshape(L*L,6)

# calculations ==================================================================

def create_mask():
    """маска в виде 3 под-решёток"""
    a = np.asarray([i % 3 for i in range(L)])
    return (a + a[:, None])%3

def calc_e(st):
    st = st.reshape(L,L)
    """calculate energy per site
        # expland state matrix"""
    a = np.concatenate((st[L-1].reshape(1,L), st, st[0].reshape(1,L)), axis=0) 
    b = np.concatenate((a[:,-1].reshape(L+2,1),a,a[:,0].reshape(L+2,1)), axis=1)
    return -np.sum(b[1:-1, 1:-1]*b[2:, 2:]*(b[2:, 1:-1]+b[1:-1, 2:]))/(L*L)  

def calc_ms(st):
    """magnetization"""
    st = st.reshape(L,L)
    msr = np.array([np.sum(st[mask==i]) for i in [0,1,2]])/(L*L)
    return np.sqrt(np.sum(msr*msr))

# model ====================================================================

def gen_state():
    """generate random init. state with lenght L*L and q=[-1,1]"""
    return np.array([np.random.choice([-1,1]) for _ in range(L*L)], dtype=np.int32)
        
def model(T,N_avg=10,N_mc=10,Relax=10):
    """Моделируем АТ"""
    E, M = [], []

    state = gen_state()
    nei = get_neigh()
    
    #relax $Relax times be4 AVG
    for __ in range(Relax):
            cy.mc_step(state, nei, T)
    #AVG every $N_mc steps
    for _ in range(N_avg):
        for __ in range(N_mc):
            cy.mc_step(state, nei, T)
        E += [calc_e(state)]
        M += [calc_ms(state)]
    
    return E, M
        
# rest ======================================================================================

def t_range(tc):
    t_ = np.array([0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3])
    t_low = np.round(-t_*tc+tc, 3)                          #low
    t_high = np.round(t_*tc+tc, 3)                      #high
    t = np.concatenate((t_low, t_high), axis=None)
    t.sort()
    return t


if __name__ == '__main__':

    global L, mask
    L = 100
    mask = create_mask()

    seed = 1
    np.random.seed(seed)      # np.random.seed(int(sys.argv[1]))

    N_avg = 50000
    N_mc = 20
    Relax = 100000

    tc = 1/(np.log(2**0.5+1)/2)       # 2.269185314213022
    t = t_range(tc)


    df_e,df_m =[pd.DataFrame() for i in range(2)]
    st = time.time()
    for ind,T in enumerate(t):
        e,m = model(T,N_avg,N_mc,Relax)
        df_e.insert(ind,T,e, True)
        df_m.insert(ind,T,m, True)
 
    title = 'bw_L'+str(L)+'_avg'+str(N_avg)+'_mc'+str(N_mc)+'_after'+str(Relax)+'mc_'
    df_e.to_csv('export/e_'+title+'seed'+str(seed)+'.csv', index = None, header=True)
    df_m.to_csv('export/m_'+title+'seed'+str(seed)+'.csv', index = None, header=True)
    print('im done in ',time.time()-st)