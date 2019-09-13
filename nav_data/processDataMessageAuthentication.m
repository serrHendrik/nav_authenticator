clear all; close all; clc;
logDir = '.\1DAY 10-09-2019 gbrm\';
logFiles = dir(logDir);
k = 1;

% Collect all ephemerids
eph = [];
for nf = 3:length(logFiles)
    logFile = logFiles(nf).name;
    [eph] = read_navMes([logDir, logFile], eph);
end

%% Simulation settings 

%%

MAXGALSVID = 36;
DeltaTimeServer = 2*3600; % sec

% Process data per satellite
nsv = 0;
for svId=1:MAXGALSVID
    svIdStr = sprintf('%2d', svId);
    svIdStr = ['E', strrep(svIdStr, ' ', '0')];
    % only visible satellites
    if (isfield(eph, svIdStr))
        nsv = nsv + 1;
        out1.ToW = [];
        out1.sp = [];
        out1.sv = [];
        out1.svCb = [];
        out1.svId = svId;
        out1.iode = [];
        out1.toe = [];
        out1.toc = [];
        
        out2.ToW = [];
        out2.sp = [];
        out2.sv = [];
        out2.svCb = [];
        out2.svId = svId;
        out2.iode = [];
        out2.toe = [];
        out2.toc = [];
        
       ephTmp = getfield(eph, svIdStr);
       idx = find((ephTmp.codes == 521) &... % 521 for INAV
                  (ephTmp.svhealth == 0));
       % check time available
       nData = length(idx);
       for j=1:nData
           
           ed = convertEph(ephTmp, idx(j));
%            if (j<nData)
%             edn = convertEph(ephTmp, idx(j+1));
%            else
%                edn = ed;
%            end
           
           % for the actual broadcast
           ToW = ed.t_oe:(ed.t_oe+600-1);%edn.t_oe-1;
%            ToW = ToW((ToW-ed.t_oe)<4*3600);
           
           [sp, sv] = satposvel_e(ed, ToW);
           [svCb] = satcb_e(ed, ToW);
            
           out1.ToW = [out1.ToW, ToW];
           out1.sp = [out1.sp, sp];
           out1.sv = [out1.sv, sv];
           out1.svCb = [out1.svCb, svCb];
           out1.iode = [out1.iode, ed.iode*ones(size(svCb))];
           out1.toe = [out1.toe, ed.t_oe*ones(size(svCb))];
           out1.toc = [out1.toc, ed.toc*ones(size(svCb))];

           % for the delayed reference 
           ToW = (ed.t_oe+DeltaTimeServer):(ed.t_oe+600-1+DeltaTimeServer);
%            ToW = ToW((ToW-ed.t_oe)<4*3600);
           
           [sp, sv] = satposvel_e(ed, ToW);
           [svCb] = satcb_e(ed, ToW);
            
           out2.ToW = [out2.ToW, ToW];
           out2.sp = [out2.sp, sp];
           out2.sv = [out2.sv, sv];
           out2.svCb = [out2.svCb, svCb];
           out2.iode = [out2.iode, ed.iode*ones(size(svCb))];
           out2.toe = [out2.toe, ed.t_oe*ones(size(svCb))];
           out2.toc = [out2.toc, ed.toc*ones(size(svCb))];
      end
       
      % Combine and compute deltas
      [C,IA,IB] = intersect(out1.ToW, out2.ToW);
      out(nsv).svId = svId;
      out(nsv).ToW = C; % time of week in sec.
      out(nsv).sp = out1.sp(:,IA)-out2.sp(:,IB); % m
      out(nsv).sv = out1.sv(:,IA)-out2.sv(:,IB); % m/s
      out(nsv).svCb = out1.svCb(:,IA)-out2.svCb(:,IB); % s
      out(nsv).iode = out1.iode(:,IA)-out2.iode(:,IB); % int number
      out(nsv).toe = out1.toe(:,IA)-out2.toe(:,IB); % s
      out(nsv).toc = out1.toc(:,IA)-out2.toc(:,IB); % s
      
    end
end

logDir=strrep(logDir, '\', '');
logDir=strrep(logDir, '.', '')
save([logDir, '.mat']);

figure,plot(out(2).toe);
figure,plot(out(2).sp.');
figure,plot(out(nsv).iode);
figure,plot(out(nsv).toe);
