%**************************************************************************
% @par read_navMes.m
% @par According to ESA FAX ESA-DTEN-NG-FAX/0016847, the Agency authorizes  
%      GMV, as TGVF-FOC contractor, to use, copy and/or modify this source   
%      code forthe performance of ESTEC Contract No. 4000108586/13/NL/NA. 
%      EU/EC Proprietary information. Unauthorized distribution, 
%      dissemination or disclosure not allowed.
% @par Project: TGVF
% @par Code Management Tool File Version: $Revision$
% @par Date (YY/MM/DD): $Date$
% @par Module: DPAF
% @par Matlab Version: Matlab r2013a
% @par Author: GMV
% @par History: see SVN History
%*************************************************************************/
function  [nav]=read_navMes(file,nav)

% Read Navigation Message
% 25/11/2010: modified to deal with NAV files in 3.0 format


fid=fopen(file, 'r');
filetype = lower(file(end));
if strcmp(filetype,'n'), sys = 'G'; end
if strcmp(filetype,'l'), sys = 'E'; end

% file version
line = fgetl(fid);
fver = line(6:9);
IONOCORR = [];

fprintf('  Reading Navigation Message  %s\n', file);
leap_seconds = 0;
% nav = {};
while 1
    line = fgetl(fid);
    if ~ischar(line), break, end 
    if strfind(line,'END OF HEADER'),break,  end % when'END OF HEADER' found, strfind is a number > 0
    
    
    line(length(line)+1:80) = ' ';                      % fill blank spaces
    data = line( 1:60);                                 % read information
    type = line(61:80);                                 % read coment
    switch type
        case 'COMMENT             '
        case 'ION ALPHA           ' % OLD
        case 'ION BETA            ' % OLD
        case 'IONOSPHERIC CORR    ' 
            % Ionospheric correction parameters
            %   types: 
            %       GAL (ai0 - ai2); 
            %       GPSA (alpha0 - alpha3); 
            %       GPSB (beta0 - beta3)
            %   parameters:
            %       GPS: alpha0-alpha3 or beta0-beta3
            %       GAL: ai0, ai1, ai2, zero
            TYPE = sscanf (data(1:5),'%s');
            IONOCORR.(TYPE).PARAM = sscanf(data(6:end),'%12e%12e%12e%12e');
        case 'DELTA-UTC: A0,A1,T,W' % OLD
        case 'TIME SYSTEM CORR    ' 
            % Corrections to transform the system time to UTC or other time systems
            %   types: 
            %       GAUT= GAL to UTC a0, a1
            %       GPUT = GPS to UTC a0, a1
            %       GLUT = GLO to UTC a0=TauC, a1=zero
            %       GPGA = GPS to GAL a0=A0G, a1=A1G
            %       GLGP = GLO to GPS a0=TauGPS, a1=zero
            TYPE = sscanf (data(1:5),'%s');
            TIMESYSCORR.(TYPE).COEF = sscanf (data(6:38),'%17e%16e');
            TIMESYSCORR.(TYPE).TREF = sscanf (data(39:45),'%7d');
            TIMESYSCORR.(TYPE).WEEK = sscanf (data(46:50),'%5d');
            %5s %2d ');
        case 'LEAP SECONDS        ' % Number of leap seconds since 6-Jan-1980
            leap_seconds = sscanf (data( 1: 6),'%f');
    end
    
end

%store iono parameters
if ~isempty(IONOCORR) && sys=='E'
    nav.ai0=IONOCORR.GAL.PARAM(1);
    nav.ai1=IONOCORR.GAL.PARAM(2);
    nav.ai2=IONOCORR.GAL.PARAM(3);
end

% Read Navigation Message in a structur
i=1;
% for j = 1 : length(file)
    while 1
        
        line= fgetl(fid);
        if line == -1, break, end
        % Save data of the satellite   
        if strcmp(fver(1),'2') % only GPS
            satid = ['G',line(1:2)];
            if strcmp(satid(2),' '), satid(2)='0'; end
        else
            satid = line(1:3);
        end
           
        if isfield(nav,satid), k = size(nav.(satid).year,2) + 1; else k = 1; end
        

        % Save data of the satellite   
        if strcmp(fver(1),'2') % only GPS
            
            int1 = (4:22);
            int2 = (23:41);
            int3 = (42:60);
            int4 = (61:79);
            nav.(satid).year(1,k) = 2000 + str2num(line(4:5));
            nav.(satid).month(1,k) = str2num(line(7:8));
            nav.(satid).day(1,k) = str2num(line(10:11));
            nav.(satid).hour(1,k) = str2num(line(13:14));
            nav.(satid).minute(1,k) = str2num(line(16:17));
            nav.(satid).second(1,k) = str2num(line(19:22));
        else
            
            int1 = (5:23);
            int2 = (24:42);
            int3 = (43:61);
            int4 = (62:80);
            nav.(satid).year(1,k) = str2num(line(5:8));
            nav.(satid).month(1,k) = str2num(line(10:11));
            nav.(satid).day(1,k) = str2num(line(13:14));
            nav.(satid).hour(1,k) = str2num(line(16:17));
            nav.(satid).minute(1,k) = str2num(line(19:21));
            nav.(satid).second(1,k) = str2num(line(22:23));
        end
           
            
%             nav(satid) = line(1:3);
            

            nav.(satid).af0(1,k) = str2num(line(int2));                    % SV clock bias (seconds)
            nav.(satid).af1(1,k) = str2num(line(int3));                    % SV clock drift (sec/sec)
            nav.(satid).af2(1,k) = str2num(line(int4));                    % SV clock drift rate (sec/sec2)
            
            line = fgetl(fid);               
            nav.(satid).iode(1,k) = str2num(line(int1));                    % IODE Issue of Data, ephemeris
            nav.(satid).C_rs(1,k) = str2num(line(int2));                   % Crs (meters)
            nav.(satid).delta_n(1,k) = str2num(line(int3));                % Delta n (rad/sec)
            nav.(satid).M_0(1,k) = str2num(line(int4));                    % M0 (rad)
            
            line = fgetl(fid);
            nav.(satid).C_uc(1,k) = str2num(line(int1));                    % Cuc (rad)
            nav.(satid).e(1,k) = str2num(line(int2));                      % eccentricity
            nav.(satid).C_us(1,k) = str2num(line(int3));                   % Cus (rad)
            nav.(satid).radA(1,k) = str2num(line(int4));                   % sqrt(A) (sqrt(m))
            nav.(satid).A(1,k)=nav.(satid).radA(1,k)^2; 
            
            line=fgetl(fid);
            nav.(satid).t_oe(1,k) = str2num(line(int1));                   % Toe Time of ephemeris (sec of GPS week)
            nav.(satid).C_ic(1,k) = str2num(line(int2));                   % Cic (rad)
            nav.(satid).OMEGA_0(1,k) = str2num(line(int3));                % OMEGA (rad)
            nav.(satid).C_is(1,k) = str2num(line(int4));                   % CIS (rad)
            
            line = fgetl(fid);
            nav.(satid).i_0(1,k) =  str2num(line(int1));                    % i0 (rad)
            nav.(satid).C_rc(1,k) = str2num(line(int2));                   % Crc (m)
            nav.(satid).omega(1,k) = str2num(line(int3));                  % omega (rad)
            nav.(satid).OMEGA_dot(1,k) = str2num(line(int4));              % OMEGA dot (rad/sec)
            
            line = fgetl(fid);
            nav.(satid).I_dot(1,k) = str2num(line(int1));                   % IDOT (rad/sec)
            nav.(satid).codes(1,k) = str2num(line(int2));                  % codes on L2 channel
            nav.(satid).weekno(1,k) = str2num(line(int3));                 % GPS week num (to go with TOE) continuous number
%             nav.(satid).pflag = str2num(line(int4)); % L2P data flag
            
            line = fgetl(fid);
            nav.(satid).svaccuracy(1,k) = str2num(line(int1));              % SV accuracy (m)
            nav.(satid).svhealth(1,k) = str2num(line(int2));               % SV health
            nav.(satid).tgd(1,k) = str2num(line(int3));                    % TGD (sec)
            nav.(satid).iodc(1,k) = str2num(line(int4));                   % IODC issue of data, clock
            
            line = fgetl(fid);
            nav.(satid).tom(1,k) = str2num(line(int1));                     % transmission time of message (sec of GPS week)
            %line = fgetl(fid);
            
             % save header info
            nav.(satid).leapsec = leap_seconds;
            
            FLAG=1;
             i=i+1;
        
    end
% end

% frewind(fid);
fclose('all');
end


