# Formula cheatsheet

$$P_d=\eta_d^PP_d^{peak},\qquad B_d=\eta_d^BB_d^{peak}$$

$$F_d=x_dF_p+F_d^m,\quad Q_d=x_dQ_p+Q_d^m,\quad R_d=x_dR_p+R_d^m$$

$$R_d\le C_d$$

$$T_d=\max(F_d/P_d,Q_d/B_d)+\ell_d+T_d^{transfer}$$

$$T_{parallel}=\max_dT_d$$

$$T_{total}=T_{serial}+T_{parallel}+\tau_{sync}$$

$$\Theta=1/T_{total}$$

$$J(x)=\lambda_TT_{total}+\lambda_EE_{total}$$

$$A(c)=1-\frac{(\sum c_i)^2}{n\sum c_i^2}$$

$$S=T_b/T_o,\qquad R_T=1-T_o/T_b$$

$$H_{prefetch}\ge\lceil T_{load}/\tau_{stage}\rceil$$

See the root README for definitions, derivations and constraints.
