#!/usr/bin/env python3
# Hochpräzise Hackfort-Diskrepanz Δ_H(L) = λ_min(W_L)  (LINDBLAD-RIEMANN A.6)
# Monom-Basis u^a auf [0,L], alle Autokorrelationen exakt als Polynome in x,
# verallgemeinertes Eigenproblem W c = λ G c (G = Gram), mpmath mit hoher Präzision.
# WICHTIG: Rayleigh-Ritz -> λ_min(N) ist eine OBERE Schranke für λ_min(W_L).
# Positive Werte beweisen nichts; ein negativer Wert (bei konvergierter Präzision) widerspräche RH.
import sys, time
import mpmath as mp

mp.mp.dps = int(sys.argv[4]) if len(sys.argv) > 4 else 120

def binom(n, k): return mp.binomial(n, k)

def poly_mul(p, q):
    r = [mp.mpf(0)]*(len(p)+len(q)-1)
    for i, a in enumerate(p):
        if a == 0: continue
        for j, b in enumerate(q): r[i+j] += a*b
    return r

def poly_eval(p, x):
    s = mp.mpf(0)
    for c in reversed(p): s = s*x + c
    return s

def autocorr_polys(N, L):
    # G[a][b](x) = ∫_0^{L-x} (w+x)^a w^b dw = Σ_k C(a,k) x^{a-k} (L-x)^{k+b+1}/(k+b+1)
    pw = [[mp.mpf(1)]]  # (L-x)^m als Polynom
    for m in range(1, 2*N+2): pw.append(poly_mul(pw[-1], [L, mp.mpf(-1)]))
    G = [[None]*N for _ in range(N)]
    for a in range(N):
        for b in range(N):
            acc = [mp.mpf(0)]*(2*N+2)
            for k in range(a+1):
                c = binom(a, k)/(k+b+1)
                q = pw[k+b+1]
                for t, qt in enumerate(q):
                    acc[t + (a-k)] += c*qt
            G[a][b] = acc
    # symmetrisieren
    S = [[[ (G[a][b][t]+G[b][a][t])/2 for t in range(2*N+2)] for b in range(N)] for a in range(N)]
    return S

def mangoldt(n):
    for p in range(2, n+1):
        if n % p == 0:
            m = n
            while m % p == 0: m //= p
            return mp.log(p) if m == 1 else mp.mpf(0)
    return mp.mpf(0)

def build(N, L):
    L = mp.mpf(L)
    S = autocorr_polys(N, L)
    Gram = mp.matrix(N, N)
    for a in range(N):
        for b in range(N): Gram[a, b] = L**(a+b+1)/(a+b+1)
    # Pol-Momente
    am = [mp.quad(lambda u: u**k*mp.exp(-u/2), [0, L]) for k in range(N)]
    bm = [mp.quad(lambda u: u**k*mp.exp(u/2), [0, L]) for k in range(N)]
    # Archimedische Momente: K(x) = 2/(1-e^{-2x})
    K = lambda x: 2/(-mp.expm1(-2*x))
    J = mp.quad(lambda x: (mp.exp(-2*x)-mp.exp(-x/2))*K(x), [0, L])
    deg = 2*N+2
    Mk = [None] + [mp.quad(lambda x, k=k: x**k*mp.exp(-x/2)*K(x), [0, L]) for k in range(1, deg)]
    c0 = -(mp.euler + mp.log(mp.pi)) - mp.log(-mp.expm1(-2*L))
    W = mp.matrix(N, N); A = mp.matrix(N, N)
    primes = []
    n = 2
    while mp.log(n) < L:
        lam = mangoldt(n)
        if lam > 0: primes.append((mp.log(n), 2*lam/mp.sqrt(n)))
        n += 1
    for a in range(N):
        for b in range(a, N):
            s = S[a][b]; g0 = s[0]
            # ∫ [g0 e^{-2x} - e^{-x/2} s(x)] K dx = g0*J + ∫ -e^{-x/2}(s(x)-g0) K dx
            arch = c0*g0 + g0*J - sum(s[k]*Mk[k] for k in range(1, len(s)) if s[k] != 0)
            pol = am[a]*bm[b] + bm[a]*am[b]
            pr = -sum(wgt*poly_eval(s, x) for x, wgt in primes)
            W[a, b] = W[b, a] = pol + arch + pr
            A[a, b] = A[b, a] = pol + arch
    return W, A, Gram, am, bm, len(primes)

def gen_eig_min(W, Gram, cons=None):
    # λ_min von W c = λ G c, optional unter Nebenbedingungen cons (Liste von Vektoren v: v·c = 0)
    N = W.rows
    Lc = mp.cholesky(Gram)
    Li = mp.inverse(Lc)
    Wt = Li * W * Li.T
    if cons:
        # Nebenbedingung v·c = 0 mit c = L^{-T} y  ->  (L^{-1} v)·y = 0
        vs = [Li * mp.matrix(v) for v in cons]
        # Orthonormalbasis des Komplements
        basis = []
        for v in vs:
            for q in basis: v = v - q*(q.T*v)[0]
            basis.append(v/mp.norm(v))
        comp = []
        for k in range(N):
            e = mp.matrix(N, 1); e[k] = 1
            for _ in range(2):
                for q in basis+comp: e = e - q*(q.T*e)[0]
            nn = mp.norm(e)
            if nn > mp.mpf(10)**(-mp.mp.dps//3): comp.append(e/nn)
            if len(comp) == N-len(basis): break
        P = mp.matrix(N, len(comp))
        for j, q in enumerate(comp):
            for i in range(N): P[i, j] = q[i]
        Wt = P.T * Wt * P
    ev = mp.eigsy(Wt, eigvals_only=True)
    return min(ev)

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    Lmax = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    print("L,e^L,primpotenzen,N,dps,lambda_min_voll,lambda_min_polfrei,lambda_min_polfrei_ohne_primzahlen", flush=True)
    for s in range(1, steps+1):
        L = Lmax*s/steps; t0 = time.time()
        W, A, Gram, am, bm, npr = build(N, L)
        lf = gen_eig_min(W, Gram)
        lr = gen_eig_min(W, Gram, [am, bm])
        l0 = gen_eig_min(A, Gram, [am, bm])
        print(f"{L:.4f},{mp.e**L:.3f},{npr},{N},{mp.mp.dps},{mp.nstr(lf,8)},{mp.nstr(lr,8)},{mp.nstr(l0,8)}", flush=True)
        print(f"# {time.time()-t0:.1f}s", file=sys.stderr, flush=True)
