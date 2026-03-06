For the homogeneous two-stream initial condition used in \cref{fig:panel3d,fig:linearized_symmetry}, the exact Landau dynamics is axisymmetric about the streaming axis: if initially
$f(\mathbf v,0)=f(v_x,v_\perp,0)$ with $v_\perp=\sqrt{v_y^2+v_z^2}$, then the $v_x=0$ slice should remain circular in the $(v_y,v_z)$ plane.
To test whether the discrete operator preserves this property, we define
$g(v_y,v_z)=f(0,v_y,v_z)$ and its angular-average defect
$\delta g=g-\langle g\rangle_\theta(r)$ at fixed $r=\sqrt{v_y^2+v_z^2}$, together with the scalar axisymmetry error
$E_{\rm ang}=\left[\int (\delta g)^2\,dA \big/ \int \langle g\rangle_\theta^2\,dA\right]^{1/2}$.
\Cref{fig:linearized_symmetry} shows that, in the present truncated Cartesian Hermite representation, the nonlinear discrete RHS $Q(f_0,f_0)$ becomes progressively more axisymmetric as $n_{\max}$ increases, whereas the linearized RHS
$L_M[h_0]=Q(h_0,M)+Q(M,h_0)$ exhibits a substantially larger symmetry defect.
The origin of this effect is not physical: the continuum linearized Landau operator about an isotropic Maxwellian is itself rotationally invariant.
Rather, the defect is introduced by the discrete background Maxwellian $M$ built from the low-order invariants, which is not exactly rotationally invariant after projection onto the finite Cartesian Hermite basis.
This is confirmed by the control curve obtained by replacing the matched projected background with the basis Maxwellian, which substantially reduces the symmetry defect.
Therefore, even when the linearized operator retains the correct conservation laws and entropy decay, it can introduce spurious symmetry breaking at finite resolution; in this representation, this provides an additional practical argument for retaining the nonlinear collision operator in far-from-equilibrium calculations.



\caption{\label{fig:linearized_symmetry}
Discrete symmetry breaking of the linearized Landau operator in a Cartesian Hermite representation.
The initial condition is the homogeneous two-stream state, which is axisymmetric about $v_x$, so the exact $v_x=0$ slice should remain circular in the $(v_y,v_z)$ plane.
Top row: non-axisymmetric defect
$\delta g(v_y,v_z)=g(v_y,v_z)-\langle g\rangle_\theta(r)$
for the matched background Maxwellian $M$, the nonlinear discrete RHS $Q(f_0,f_0)$, and the linearized discrete RHS $L_M(h_0)=Q(h_0,M)+Q(M,h_0)$, shown for $n_{\max}=12$.
Bottom: axisymmetry error
$E_{\rm ang}=\left[\int (\delta g)^2\,dA \big/ \int \langle g\rangle_\theta^2\,dA\right]^{1/2}$
versus Hermite resolution.
The matched projected background is itself not exactly rotationally invariant, and the linearized operator amplifies this discrete defect much more strongly than the nonlinear operator.
A control linearization about the basis Maxwellian reduces the error, identifying the projected background---rather than the continuum linearization itself---as the main source of symmetry breaking.
Thus, at finite truncation, linearization can preserve conservation and entropy diagnostics while still distorting rotational structure.}
