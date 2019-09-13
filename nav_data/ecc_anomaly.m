% /************************************************************************
% *
% * Function: Calculate eccentric anomaly
% *
% *************************************************************************/
function [Ek] = ecc_anomaly(Mk, e)

Ek = Mk;
i = 0;
new_Ek = Mk + (e * sin(Ek));
Ek_diff = abs(new_Ek - Ek);
Ek = new_Ek;

while ((Ek_diff > 1.0e-8) && (i < 10))
    new_Ek = Mk + (e * sin(Ek));
    Ek_diff = abs(new_Ek - Ek);
    Ek = new_Ek;
    i = i + 1;
end
    
