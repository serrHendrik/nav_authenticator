% /************************************************************************
% *
% * Function: Evaluate SV position and velocity based on ephemeris
% *
% *************************************************************************/
function [sp, sv] = satposvel_e(ed, time)

sp = NaN(3, length(time));
sv = sp;
Ek = zeros(1, length(time));

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
        disp('warning ephemeris week');
else
    if (tk < -SECS_PER_HALF_WEEK)
        tk = tk + SECS_PER_WEEK;
        disp('warning ephemeris week');
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

% compute velocity */
for i = 1:length(time)
    vp(1) = ed.n*ed.axis*ed.axis/rk(i);
    vp(2) = vp(1)*cos_Ek(i)*ed.r1me2;
    vp(1) = vp(1)*(-sin_Ek(i));

    rzlk(1,1) = cosok(i);
    rzlk(1,2) = -sinok(i);
    rzlk(1,3) = 0.0;
    rzlk(2,1) = sinok(i);
    rzlk(2,2) = cosok(i);
    rzlk(2,3) = 0.0;
    rzlk(3,1) = 0.0;
    rzlk(3,2) = 0.0;
    rzlk(3,3) = 1.0;

    rxik(1,1) = 1.0;
    rxik(1,2) = 0.0;
    rxik(1,3) = 0.0;

    rxik(2,1) = 0.0;
    rxik(2,2) = cosik(i);
    rxik(2,3) = -sinik(i);

    rxik(3,1) = 0.0;
    rxik(3,2) = sinik(i);
    rxik(3,3) = cosik(i);

    temp = rzlk*rxik;

    cosuf = cos(uk(i) - fk(i));
    sinuf = sin(uk(i) - fk(i));

    vel = vp;
    vel(3) = 0.0;

    rzmk(1,1) = cosuf;
    rzmk(1,2) = -sinuf;
    rzmk(1,3) = 0.0;

    rzmk(2,1) = sinuf;
    rzmk(2,2) = cosuf;
    rzmk(2,3) = 0.0;

    rzmk(3,1) = 0.0;
    rzmk(3,2) = 0.0;
    rzmk(3,3) = 1.0;
    temp1 = temp*rzmk;

    sv(:, i) = temp1*vel.';
    sv(1, i) = sv(1, i) + WGS84_OMEGDOTE * sp(2, i);
    sv(2, i) = sv(2, i) - WGS84_OMEGDOTE * sp(1, i);
    
end