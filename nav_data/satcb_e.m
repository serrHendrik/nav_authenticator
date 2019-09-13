% /************************************************************************
% *
% * Function: Evaluate SV clock bias based on ephemeris
% *
% *************************************************************************/
function [dTclk_Ofset] = satcb_e(ed, time)

SECS_PER_HALF_WEEK = 3.5*3600*24;
SECS_PER_WEEK = 7*3600*24;
WGS84_SQRT_U    = 1.9964980385665296e7;
WGS84_OMEGDOTE  = 7.2921151467e-5;

%Ofset Model:                                                             *
%dTclk_Ofset=af0+af1*(T-Toc)+af2*(T-Toc)^2+.....                          *
%af :(1/Sec^i)    =>Matrix of Coeeficient for satellite offset            *
%Ttr:(Sec)        => Time of transmission                                 *
%Toc:(Sec)        => Sv Clock refernce time                               *
%dTclk_Ofset:(Sec)=> Sv Clock offset time                                 *
%**************************************************************************

% FieldsGAL = {'HS', 'IOD', 'toc',  'af0', 'af1', 'af2', 'tgd', 't_oe', 'e',...
%           'i_0', 'idot', 'm_0', 'omega', 'omega0', 'omegaDot',...
%           'rootOfA', 'deltaN', 'c_ic','c_is','c_rc','c_rs',...
%           'c_us','c_uc'};


af = [ed.af0, ed.af1, ed.af2] ;
Dim1=size(af);
Order_Coef=length(Dim1);

dTclk_Ofset=0;
T = time-ed.toc;
T(T>SECS_PER_HALF_WEEK) = T(T>SECS_PER_HALF_WEEK)-SECS_PER_WEEK;
T(T<-SECS_PER_HALF_WEEK) = T(T<-SECS_PER_HALF_WEEK)+SECS_PER_WEEK;

for i=1:Order_Coef
    dTclk_Ofset=dTclk_Ofset+af(i)*T.^(i-1);
end
% PC add the BGD as af0
dTclk_Ofset=dTclk_Ofset-ed.tgd;
