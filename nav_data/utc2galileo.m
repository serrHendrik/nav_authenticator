function [ToW, WN] = utc2galileo(date0)
%UTC2GALILEO Convert UTC(GMT) time tags to GPS time accounting for leap seconds
%   UTC2GALILEO(date) corrects an array of UTC dates(in any matlab format) for
%   leap seconds and returns an array of Galileo datenums where:
%   Galileo = UTC + steptime
%   Currently step times are through Jan 1 2009, but need to be added below
%   as they are instuted. All input dates must be later than the start of
%   the GST (Galileo System Time) 0h UTC on Sunday, 22 August 1999 
%   (midnight between 21 and 22 August).


%% ADD NEW LEAP DATES HERE:
stepdates = [...
    'Aug 22 1999'
    'Jan  1 2006'
    'Jan  1 2009'
    'Jul  1 2012'
    'Jul  1 2015'
    'Jan  1 2016'];

%% Convert Steps to datenums and make step offsets
stepdates = datenum(stepdates)'; %step date coversion
steptime = (0:length(stepdates)-1)'./86400;  %corresponding step time (sec)

%% Arg Checking
if ~isnumeric(date0) %make sure date0 are datenums, if not try converting
    date0 = datenum(date0); %will error if not a proper format
end

if ~isempty(find(date0 < stepdates(1)))%date0 must all be after GPS start date
    error('Input dates must be after 00:00:00 on Jan 6th 1980') 
end

%% Array Sizing
sz = size(date0);
date0 = date0(:);

date0 = repmat(date0,[1 size(stepdates,2)]);
stepdates = repmat(stepdates,[size(date0,1) 1]);

%% Conversion
delta = steptime(sum((date0 - stepdates) >= 0,2));
date1 = date0(:,1) + delta;

%% Reshape Output Array
date1 = reshape(date1,sz);

%% Leap seconds
leapsec = datevec(delta);
leapsec = leapsec(:,end);
tmp = round((date1 - datenum(1999,8,22,0,0,0)*ones(size(date1)))*3600*24);
SEC_WEEK = 3600*24*7;
ToW = mod(tmp, SEC_WEEK);
WN = floor(tmp/SEC_WEEK);


