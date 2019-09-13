function [ed] = convertEph(ephList, j)

t0 = [ephList.second(j), ephList.minute(j), ephList.hour(j), ...
    ephList.day(j), ephList.month(j), ephList.year(j)];

t0 = datenum(fliplr(t0));
tmp = round((t0 - datenum(1999,8,22,0,0,0)*ones(size(t0)))*3600*24);
SEC_WEEK = 3600*24*7;
ed.toc = mod(tmp, SEC_WEEK);

ed.svaccuracy = ephList.svaccuracy(j);
ed.svhealth = ephList.svhealth(j);

ed.af0 = ephList.af0(j);
ed.af1 = ephList.af1(j);
ed.af2 = ephList.af2(j);

ed.t_oe = ephList.t_oe(j);
ed.iode = ephList.iode(j);
ed.iodc = ephList.iodc(j);
ed.wn = ephList.weekno(j);

ed.c_rs = ephList.C_rs(j);
ed.c_rc = ephList.C_rc(j);
ed.c_us = ephList.C_us(j);
ed.c_uc = ephList.C_uc(j);
ed.c_is = ephList.C_is(j);
ed.c_ic = ephList.C_ic(j);
ed.tgd = ephList.tgd(j);

ed.omega0 = ephList.OMEGA_0(j);
ed.m_0 = ephList.M_0(j);
ed.e = ephList.e(j);
ed.rootOfA = ephList.radA(j);
ed.i_0 = ephList.i_0(j);
ed.idot = ephList.I_dot(j);
ed.omega = ephList.omega(j);
ed.omegaDot = ephList.OMEGA_dot(j);

