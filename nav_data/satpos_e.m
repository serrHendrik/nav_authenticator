% /************************************************************************
% *
% * Function: Evaluate SV position based on ephemeris
% *
% *************************************************************************/
function [sp] = satpos_e(ed, time)

SECS_PER_HALF_WEEK = 3.5*3600*24;
SECS_PER_WEEK = 7*3600*24;
WGS84_SQRT_U    = 1.9964980385665296e7;
WGS84_OMEGDOTE  = 7.2921151467e-5;

ed.axis = ed.rootOfA^2;
if (ed.axis>0)
    ed.n = WGS84_SQRT_U / (ed.rootOfA^3);
else
    ed.n = 0.0;
end
ed.r1me2   = sqrt(abs(1.0 - ed.e^2));
ed.omega_n = ed.omega0 - (WGS84_OMEGDOTE * ed.t_oe);
ed.odot_n  = ed.omegaDot - WGS84_OMEGDOTE;

tk = time - ed.t_oe;
if (tk > SECS_PER_HALF_WEEK)
        tk = tk - SECS_PER_WEEK;
else
    if (tk < -SECS_PER_HALF_WEEK)
        tk = tk + SECS_PER_WEEK;
    end
end
Mk = ed.m_0 + (ed.n * tk);
for i =1:length(Mk)
    Ek(i) = ecc_anomaly(Mk(i), ed.e);
end
cos_Ek = cos(Ek);
sin_Ek = sin(Ek);
arg_y = ed.r1me2 * sin_Ek;
arg_x = cos_Ek - ed.e;
fk = atan2(arg_y, arg_x);
phik = fk + ed.omega;
del_tr = phik + phik;
sin_phik2 = sin(del_tr);
cos_phik2 = cos(del_tr);
del_uk = (ed.c_us * sin_phik2) + (ed.c_uc * cos_phik2);
del_rk = (ed.c_rc * cos_phik2) + (ed.c_rs * sin_phik2);
del_ik = (ed.c_ic * cos_phik2) + (ed.c_is * sin_phik2);

rk = ed.axis * (1.0 - (ed.e * cos_Ek)) + del_rk;
ik = ed.i_0 + (ed.idot * tk) + del_ik;
ok = ed.omega_n + (ed.odot_n * tk);

uk = phik + del_uk;
xkp = rk.*cos(uk);
ykp = rk.*sin(uk);
sinok = sin(ok);
cosok = cos(ok);
sinik = sin(ik);
cosik = cos(ik);

% compute position
sp(1,:) = (xkp.*cosok) - (ykp.*cosik.*sinok);
sp(2,:) = (xkp.*sinok) + (ykp.*cosik.*cosok);
sp(3,:) = ykp.*sinik;

