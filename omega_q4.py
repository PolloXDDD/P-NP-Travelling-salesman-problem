
import math, time, argparse
import numpy as np

class DSU:
    __slots__ = ("p","sz")
    def __init__(self,n):
        self.p=list(range(n)); self.sz=[1]*n
    def find(self,x):
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]
            x=self.p[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a==b: return False
        if self.sz[a]<self.sz[b]: a,b=b,a
        self.p[b]=a; self.sz[a]+=self.sz[b]
        return True

def tour_length(D,tour):
    n=len(tour)
    return float(sum(D[tour[i],tour[(i+1)%n]] for i in range(n)))

def _cycle_from_edges(n,edges):
    adj=[[] for _ in range(n)]
    for u,v in edges:
        adj[u].append(v); adj[v].append(u)
    if any(len(x)!=2 for x in adj):
        raise RuntimeError("not 2-regular")
    tour=[0]; prev=-1; cur=0
    for _ in range(n-1):
        a,b=adj[cur]
        nxt=a if a!=prev else b
        tour.append(nxt); prev,cur=cur,nxt
    if tour[-1] not in adj[0]:
        raise RuntimeError("multiple cycles")
    return tour

def _one_full_two_opt_pass(D,tour):
    n=len(tour); best_delta=0.0; best=None
    for i in range(n-1):
        a=tour[i]; b=tour[(i+1)%n]
        for j in range(i+2, n if i>0 else n-1):
            c=tour[j]; d=tour[(j+1)%n]
            delta=D[a,c]+D[b,d]-D[a,b]-D[c,d]
            if delta<best_delta:
                best_delta=float(delta); best=(i+1,j)
    if best:
        l,r=best
        tour=tour[:l]+tour[l:r+1][::-1]+tour[r+1:]
    return tour

def omega_q2(D,beta=0.5,two_opt_pass=True):
    D=np.asarray(D,dtype=np.float64)
    n=D.shape[0]
    if D.shape!=(n,n) or n<3: raise ValueError("D must be n x n, n>=3")
    W=D.copy(); np.fill_diagonal(W,np.inf)
    two=np.argpartition(W,kth=1,axis=1)[:,:2]
    vals=np.take_along_axis(W,two,axis=1)
    ord2=np.argsort(vals,axis=1)
    first_idx=two[np.arange(n),ord2[:,0]]
    first_val=vals[np.arange(n),ord2[:,0]]
    second_val=vals[np.arange(n),ord2[:,1]]

    iu,iv=np.triu_indices(n,1)
    w=D[iu,iv]
    alt_u=np.where(first_idx[iu]==iv,second_val[iu],first_val[iu])
    alt_v=np.where(first_idx[iv]==iu,second_val[iv],first_val[iv])
    score=w-beta*((alt_u-w)+(alt_v-w))
    order=np.argsort(score,kind="stable")

    deg=np.zeros(n,dtype=np.int8)
    dsu=DSU(n); edges=[]
    for idx in order:
        u=int(iu[idx]); v=int(iv[idx])
        if deg[u]>=2 or deg[v]>=2: continue
        if dsu.find(u)==dsu.find(v): continue
        edges.append((u,v)); deg[u]+=1; deg[v]+=1; dsu.union(u,v)
        if len(edges)==n-1: break
    if len(edges)!=n-1: raise RuntimeError("failed to form Hamiltonian path")
    ends=np.flatnonzero(deg==1)
    edges.append((int(ends[0]),int(ends[1])))
    tour=_cycle_from_edges(n,edges)
    if two_opt_pass:
        tour=_one_full_two_opt_pass(D,tour)
    return tour

def held_karp_exact(D):
    D=np.asarray(D,dtype=np.float64); n=D.shape[0]; m=n-1; size=1<<m
    inf=float("inf")
    dp=np.full((size,m),inf); parent=np.full((size,m),-1,dtype=np.int16)
    for j in range(m): dp[1<<j,j]=D[0,j+1]
    for mask in range(1,size):
        bits=mask
        while bits:
            lb=bits&-bits; j=lb.bit_length()-1; pm=mask^lb
            if pm:
                best=inf; bestk=-1; kb=pm
                while kb:
                    klb=kb&-kb; k=klb.bit_length()-1
                    val=dp[pm,k]+D[k+1,j+1]
                    if val<best: best=val; bestk=k
                    kb^=klb
                dp[mask,j]=best; parent[mask,j]=bestk
            bits^=lb
    full=size-1; best=inf; last=-1
    for j in range(m):
        val=dp[full,j]+D[j+1,0]
        if val<best: best=val; last=j
    rev=[]; mask=full; j=last
    while j>=0:
        rev.append(j+1); pj=int(parent[mask,j]); mask^=1<<j; j=pj
    return [0]+rev[::-1],float(best)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=12)
    ap.add_argument("--seed",type=int,default=20260922)
    args=ap.parse_args()
    rng=np.random.default_rng(args.seed)
    pts=rng.random((args.n,2))
    delta=pts[:,None,:]-pts[None,:,:]
    D=np.sqrt((delta*delta).sum(axis=2))
    t0=time.perf_counter(); tour=omega_q2(D); t1=time.perf_counter()
    print("OMEGA-Q2",tour_length(D,tour),"time_ms",(t1-t0)*1000)
    if args.n<=18:
        t0=time.perf_counter(); ot,oc=held_karp_exact(D); t1=time.perf_counter()
        print("EXACT",oc,"time_ms",(t1-t0)*1000)
        print("gap_%",(tour_length(D,tour)/oc-1)*100)


def _two_opt_once(D,tour):
    n=len(tour); best_delta=0.0; pair=None
    for i in range(n-1):
        a=tour[i]; b=tour[(i+1)%n]
        for j in range(i+2,n if i>0 else n-1):
            c=tour[j]; d=tour[(j+1)%n]
            delta=D[a,c]+D[b,d]-D[a,b]-D[c,d]
            if delta<best_delta:
                best_delta=float(delta); pair=(i+1,j)
    if pair is not None:
        l,r=pair
        return tour[:l]+tour[l:r+1][::-1]+tour[r+1:]
    return tour

def omega_q2_plus(D,betas=(0.0,0.25,0.5,1.0,2.0),two_opt_passes=4):
    best=None; best_cost=float("inf")
    for beta in betas:
        t=omega_q2(D,beta=beta,two_opt_pass=False)
        c=tour_length(D,t)
        for _ in range(two_opt_passes):
            nt=_two_opt_once(D,t)
            nc=tour_length(D,nt)
            if nc>=c-1e-12: break
            t,c=nt,nc
        if c<best_cost:
            best,best_cost=t,c
    return best


# ============================================================
# OMEGA-Q2 STRICT PORTFOLIO
# All auxiliary search passes below are O(n^2).
# Current overall bound: O(n^2 log n), because omega_q2()
# comparison-sorts Theta(n^2) edge scores.
# ============================================================

def _normalize_tour0(t):
    k=t.index(0)
    return t[k:]+t[:k]

def _two_opt_once_q2(D,tour):
    n=len(tour); best_delta=0.0; best=None
    for i in range(n-1):
        a=tour[i]; b=tour[(i+1)%n]
        for j in range(i+2, n if i>0 else n-1):
            c=tour[j]; d=tour[(j+1)%n]
            delta=D[a,c]+D[b,d]-D[a,b]-D[c,d]
            if delta<best_delta-1e-15:
                best_delta=float(delta); best=(i+1,j)
    if best is None: return tour,False
    l,r=best
    return tour[:l]+tour[l:r+1][::-1]+tour[r+1:],True

def _relocate_once_q2(D,tour):
    n=len(tour); best_delta=0.0; best=None
    for i in range(n):
        p=tour[(i-1)%n]; x=tour[i]; q=tour[(i+1)%n]
        removal=D[p,q]-D[p,x]-D[x,q]
        for j in range(n):
            if j==i or j==(i-1)%n: continue
            a=tour[j]; b=tour[(j+1)%n]
            if b==x: continue
            delta=removal+D[a,x]+D[x,b]-D[a,b]
            if delta<best_delta-1e-15:
                best_delta=float(delta); best=(i,j)
    if best is None: return tour,False
    i,j=best; x=tour[i]; t=tour[:]; t.pop(i)
    if i<j: j-=1
    t.insert(j+1,x)
    return t,True

def _swap_once_q2(D,tour):
    n=len(tour); best_delta=0.0; best=None
    def after(k,i,j):
        if k==i: return tour[j]
        if k==j: return tour[i]
        return tour[k]
    for i in range(1,n):
        for j in range(i+1,n):
            affected={(i-1)%n,i,(j-1)%n,j}
            old=0.0; new=0.0
            for e in affected:
                f=(e+1)%n
                old+=D[tour[e],tour[f]]
                new+=D[after(e,i,j),after(f,i,j)]
            delta=float(new-old)
            if delta<best_delta-1e-15:
                best_delta=delta; best=(i,j)
    if best is None: return tour,False
    t=tour[:]; i,j=best; t[i],t[j]=t[j],t[i]
    return t,True

def _polish_q2(D,tour,rounds=8):
    t=tour[:]
    for _ in range(rounds):
        changed=False
        t,ch=_two_opt_once_q2(D,t); changed|=ch
        t,ch=_relocate_once_q2(D,t); changed|=ch
        t,ch=_swap_once_q2(D,t); changed|=ch
        if not changed: break
    return t

def _nearest_neighbor_start_q2(D,start):
    n=len(D); used=np.zeros(n,dtype=bool); used[start]=True
    t=[int(start)]; cur=int(start)
    for _ in range(n-1):
        row=D[cur].copy(); row[used]=np.inf
        nxt=int(np.argmin(row))
        t.append(nxt); used[nxt]=True; cur=nxt
    return _normalize_tour0(t)

def _insertion_q2(D,seed=0,mode="farthest"):
    n=len(D)
    unused=np.ones(n,dtype=bool); unused[seed]=False
    if mode=="farthest":
        partner=int(np.argmax(np.where(unused,D[seed],-np.inf)))
    else:
        partner=int(np.argmin(np.where(unused,D[seed],np.inf)))
    tour=[seed,partner]; unused[partner]=False
    mind=np.minimum(D[:,seed],D[:,partner])
    mind[~unused]=np.inf if mode=="nearest" else -np.inf
    while len(tour)<n:
        if mode=="farthest":
            x=int(np.argmax(np.where(unused,mind,-np.inf)))
        else:
            x=int(np.argmin(np.where(unused,mind,np.inf)))
        best_inc=float("inf"); best_pos=1
        m=len(tour)
        for i in range(m):
            a=tour[i]; b=tour[(i+1)%m]
            inc=D[a,x]+D[x,b]-D[a,b]
            if inc<best_inc:
                best_inc=float(inc); best_pos=i+1
        tour.insert(best_pos,x); unused[x]=False
        for u in range(n):
            if unused[u] and D[u,x]<mind[u]:
                mind[u]=D[u,x]
        mind[x]=np.inf if mode=="nearest" else -np.inf
    return _normalize_tour0(tour)

def _fixed_kicks_q2(tour):
    n=len(tour)
    if n<8: return []
    cutsets=[]
    for shift in (0,1,2):
        a=max(1,n//5+shift)
        b=max(a+1,2*n//5+shift)
        c=max(b+1,3*n//5+shift)
        d=max(c+1,4*n//5+shift)
        if d<n: cutsets.append((a,b,c,d))
    extra=[
        (1,max(2,n//3),max(3,2*n//3),n-1),
        (max(1,n//4),max(2,n//2-1),max(3,3*n//4),n-1),
        (2,max(3,n//3+1),max(4,2*n//3+1),n-1),
    ]
    for a,b,c,d in extra:
        if 0<a<b<c<d<n: cutsets.append((a,b,c,d))
    out=[]
    for a,b,c,d in cutsets:
        A=tour[:a]; B=tour[a:b]; C=tour[b:c]; D=tour[c:d]; E=tour[d:]
        variants=[
            A+C+B+D+E,
            A+B+D+C+E,
            A+D+C+B+E,
            A+C+D+B+E,
            A+D+B+C+E,
            A+B+C[::-1]+D+E,
            A+B[::-1]+C+D+E,
            A+B+D[::-1]+C+E,
        ]
        for p in variants:
            if len(set(p))==n: out.append(_normalize_tour0(p))
    return out

def omega_q2_strict(D):
    """
    Experimental symmetric-TSP solver.

    Empirical status in the bundled Sep-2026 tests:
    exact optimum on all tested instances through n=16.

    This is NOT an exactness proof.

    Current comparison-model complexity:
        O(n^2 log n)
    from sorting Theta(n^2) edge scores in omega_q2().
    All portfolio counts and local-search rounds are fixed constants,
    and every auxiliary pass is O(n^2).
    """
    D=np.asarray(D,dtype=np.float64)
    n=len(D)
    candidates=[]
    for beta in (-1.0,-0.5,0.0,0.125,0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0):
        candidates.append(omega_q2(D,beta=beta,two_opt_pass=False))

    starts=list(range(min(n,16)))
    for s in starts:
        candidates.append(_nearest_neighbor_start_q2(D,s))
    for s in starts[:8]:
        candidates.append(_insertion_q2(D,s,"farthest"))
        candidates.append(_insertion_q2(D,s,"nearest"))

    best=None; best_cost=float("inf")
    for t in candidates:
        t=_polish_q2(D,t,rounds=8)
        c=tour_length(D,t)
        if c<best_cost:
            best,best_cost=t,c

    for kt in _fixed_kicks_q2(best)[:48]:
        t=_polish_q2(D,kt,rounds=10)
        c=tour_length(D,t)
        if c<best_cost:
            best,best_cost=t,c

    snapshot=best
    for kt in _fixed_kicks_q2(snapshot)[:24]:
        t=_polish_q2(D,kt,rounds=10)
        c=tour_length(D,t)
        if c<best_cost:
            best,best_cost=t,c
    return best


# ============================================================
# OMEGA-Q3
# Plateau-aware extension of OMEGA-Q2 STRICT.
#
# New idea:
#   permit a fixed-depth, fixed-width walk across zero-cost plateaus.
#   A chain of neutral moves is accepted only if it exposes a strict
#   improvement afterwards.
#
# Asymptotics:
#   Q2 core: O(n^2 log n) due to sorting Theta(n^2) edge scores.
#   Q3 bridge: depth, width, rounds are fixed constants; every scan
#              considers O(n^2) 2-opt/relocate/swap moves.
#   Therefore Q3 remains O(n^2 log n) in the comparison model.
# ============================================================

def _q3_normalize(t):
    t=list(t)
    k=t.index(0)
    t=t[k:]+t[:k]
    rev=[t[0]]+list(reversed(t[1:]))
    return tuple(t) if tuple(t)<=tuple(rev) else tuple(rev)

def _q3_apply_move(tour,mv):
    kind,a,b=mv
    t=list(tour)
    if kind=="2opt":
        return t[:a]+t[a:b+1][::-1]+t[b+1:]
    if kind=="relocate":
        x=t.pop(a)
        j=b
        if a<j:
            j-=1
        t.insert(j+1,x)
        return t
    if kind=="swap":
        t[a],t[b]=t[b],t[a]
        return t
    raise ValueError("unknown Q3 move")

def _q3_scan_moves(D,tour,neutral=False,neutral_cap=96):
    n=len(tour)
    best_delta=0.0
    best_move=None
    neutrals=[]

    # 2-opt
    for i in range(n-1):
        a,b=tour[i],tour[(i+1)%n]
        for j in range(i+2,n if i>0 else n-1):
            c,d=tour[j],tour[(j+1)%n]
            delta=float(D[a,c]+D[b,d]-D[a,b]-D[c,d])
            if delta<best_delta-1e-12:
                best_delta=delta
                best_move=("2opt",i+1,j)
            elif neutral and abs(delta)<=1e-12 and len(neutrals)<neutral_cap:
                neutrals.append(("2opt",i+1,j))

    # relocate
    for i in range(n):
        p=tour[(i-1)%n]
        x=tour[i]
        q=tour[(i+1)%n]
        removal=D[p,q]-D[p,x]-D[x,q]
        for j in range(n):
            if j==i or j==(i-1)%n:
                continue
            a=tour[j]
            b=tour[(j+1)%n]
            if b==x:
                continue
            delta=float(removal+D[a,x]+D[x,b]-D[a,b])
            if delta<best_delta-1e-12:
                best_delta=delta
                best_move=("relocate",i,j)
            elif neutral and abs(delta)<=1e-12 and len(neutrals)<neutral_cap:
                neutrals.append(("relocate",i,j))

    # swap with O(1) delta per pair
    def swap_delta(i,j):
        affected={(i-1)%n,i,(j-1)%n,j}
        def after(k):
            if k==i:
                return tour[j]
            if k==j:
                return tour[i]
            return tour[k]
        old=new=0.0
        for e in affected:
            f=(e+1)%n
            old+=D[tour[e],tour[f]]
            new+=D[after(e),after(f)]
        return float(new-old)

    for i in range(1,n):
        for j in range(i+1,n):
            delta=swap_delta(i,j)
            if delta<best_delta-1e-12:
                best_delta=delta
                best_move=("swap",i,j)
            elif neutral and abs(delta)<=1e-12 and len(neutrals)<neutral_cap:
                neutrals.append(("swap",i,j))

    return best_delta,best_move,neutrals

def _q3_strict_polish(D,tour,rounds=10):
    t=list(tour)
    for _ in range(rounds):
        _,mv,_=_q3_scan_moves(D,t,neutral=False)
        if mv is None:
            break
        t=_q3_apply_move(t,mv)
    return t

def _q3_plateau_bridge(D,tour,depth=2,beam_width=64,neutral_cap=96):
    """
    Search a constant-depth neutral plateau.

    A state is expanded only through zero-cost 2-opt/relocate/swap moves.
    As soon as a strict improvement becomes available, take it and repolish.

    Fixed depth and width keep the bridge O(n^2).
    """
    base=list(tour)
    base_cost=tour_length(D,base)

    beam=[(base,[])]
    seen={_q3_normalize(base)}

    for level in range(depth+1):
        next_states=[]

        for state,path in beam:
            _,mv,neutrals=_q3_scan_moves(
                D,state,
                neutral=(level<depth),
                neutral_cap=neutral_cap
            )

            if mv is not None:
                improved=_q3_apply_move(state,mv)
                improved=_q3_strict_polish(D,improved,rounds=10)
                if tour_length(D,improved)<base_cost-1e-12:
                    return improved,path+[mv]

            if level<depth:
                for nm in neutrals:
                    nt=_q3_apply_move(state,nm)
                    key=_q3_normalize(nt)
                    if key not in seen:
                        seen.add(key)
                        next_states.append((nt,path+[nm]))
                        if len(next_states)>=beam_width:
                            break
                if len(next_states)>=beam_width:
                    break

        if not next_states:
            break
        beam=next_states[:beam_width]

    return None,None

def omega_q3(D):
    """
    OMEGA-Q3 = unchanged OMEGA-Q2 STRICT + plateau bridge.

    Current comparison-model complexity: O(n^2 log n).
    No exact fallback is used.
    """
    best=omega_q2_strict(D)
    best_cost=tour_length(D,best)

    # Fixed number of bridge attempts.
    for _ in range(4):
        candidate,_=_q3_plateau_bridge(
            D,best,
            depth=2,
            beam_width=64,
            neutral_cap=96
        )
        if candidate is None:
            break

        c=tour_length(D,candidate)
        if c<best_cost-1e-12:
            best,best_cost=candidate,c
        else:
            break

    return best


# ============================================================
# OMEGA-Q4 — COMPRESSED PLATEAU CLOSURE
#
# Q4 removes Q3's fixed neutral-depth barrier.
#
# Given a tour, cut one expensive tour edge to obtain a Hamiltonian
# path. Then explore neutral Pósa rotations until a fixed point in
# the compressed state space of ordered endpoint pairs (u,v).
#
# A neutral rotation preserves the Hamiltonian-path cost exactly.
# If any reachable endpoint pair can be closed by an edge cheaper
# than the deleted edge, Q4 obtains a strictly better tour.
#
# No exact solver / Held-Karp fallback is used.
#
# Current prototype worst-case:
#   Q3 core:              O(n^2 log n)
#   pair-closure repair:  O(n^3) for a fixed number of cut edges
#
# The state space itself has only O(n^2) endpoint-pair states.
# ============================================================

from collections import deque

def _q4_rotate_right(path,i):
    return path[:i+1]+path[:i:-1]

def _q4_pair_closure_neutral(D,path,tol=1e-12):
    n=len(path)
    key0=(path[0],path[-1])
    reps={key0:path[:]}
    q=deque([key0])

    while q:
        key=q.popleft()
        cur=reps[key]
        left,right=cur[0],cur[-1]

        # Rotate the right endpoint.
        # Neutral iff the inserted edge has exactly the cost of
        # the path edge that is removed.
        for i in range(0,n-2):
            if abs(D[cur[i],right]-D[cur[i],cur[i+1]])<=tol:
                nk=(left,cur[i+1])
                if nk not in reps:
                    nxt=_q4_rotate_right(cur,i)
                    reps[nk]=nxt
                    q.append(nk)

        # Rotate the left endpoint symmetrically.
        for i in range(2,n):
            if abs(D[left,cur[i]]-D[cur[i-1],cur[i]])<=tol:
                nk=(cur[i-1],right)
                if nk not in reps:
                    rev=list(reversed(cur))
                    rp=n-1-i
                    nxt_rev=_q4_rotate_right(rev,rp)
                    nxt=list(reversed(nxt_rev))
                    reps[nk]=nxt
                    q.append(nk)

    return reps

def _q4_best_plateau_exit(D,tour,cut_cap=8,tol=1e-12):
    n=len(tour)
    C=tour_length(D,tour)

    edge_info=[]
    for i in range(n):
        a,b=tour[i],tour[(i+1)%n]
        edge_info.append((float(D[a,b]),i,a,b))
    edge_info.sort(reverse=True)

    best=list(tour)
    best_cost=C
    best_meta=None

    # Fixed cut_cap: Q4 attacks a constant number of most expensive
    # current tour edges.
    for wcut,idx,a,b in edge_info[:min(cut_cap,n)]:
        j=(idx+1)%n
        path=tour[j:]+tour[:j]
        path_cost=C-wcut

        reps=_q4_pair_closure_neutral(D,path,tol=tol)

        for (u,v),p in reps.items():
            close=float(D[u,v])
            new_cost=path_cost+close

            if new_cost<best_cost-tol:
                best=p
                best_cost=new_cost
                best_meta={
                    "cut":(a,b),
                    "cut_weight":wcut,
                    "closing":(u,v),
                    "closing_weight":close,
                    "states":len(reps),
                    "gain":C-new_cost,
                }

    return best,best_meta

def omega_q4(D):
    """
    OMEGA-Q4 = OMEGA-Q3 + compressed neutral plateau closure.

    Unlike Q3, Q4 has no neutral-depth parameter.

    Current implementation:
        O(n^3) worst-case plateau repair
    on top of Q3's O(n^2 log n) core.
    """
    best=omega_q3(D)
    best_cost=tour_length(D,best)

    # Fixed number of global closure/improvement rounds.
    for _ in range(4):
        cand,_=_q4_best_plateau_exit(D,best,cut_cap=8)
        c=tour_length(D,cand)

        if c<best_cost-1e-12:
            best,best_cost=cand,c
        else:
            break

    return best
