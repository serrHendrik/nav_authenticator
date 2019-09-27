%Script to easily convert out struct to a table.
%A table is more convenient to select the columns of interest, query based
%on conditions etc.

vars = {'svId', 'ToW', 'sp_X', 'sp_Y', 'sp_Z', 'sv_X', 'sv_Y', 'sv_Z', 'svCb', ...
    'iode', 'toe', 'toc'};
data = cell2table(cell(0,length(vars)), 'VariableNames', vars);

for i = 1:length(out)
   row = out(i);
   nb_rows = length(row.ToW);
   
   if nb_rows > 0
       svId = repmat(row.svId,nb_rows,1);
       ToW = transpose(row.ToW);
       sp_X = transpose(row.sp(1,:));
       sp_Y = transpose(row.sp(2,:));
       sp_Z = transpose(row.sp(3,:));
       sv_X = transpose(row.sv(1,:));
       sv_Y = transpose(row.sv(2,:));
       sv_Z = transpose(row.sv(3,:));
       svCb = transpose(row.svCb);
       iode = transpose(row.iode);
       toe = transpose(row.toe);
       toc = transpose(row.toc);
       
       new_rows = table(svId, ToW, sp_X, sp_Y, sp_Z, sv_X, sv_Y, sv_Z, ...
           svCb, iode, toe, toc, ...
           'VariableNames', vars);
       
       data = [data; new_rows];
       
   end
end

%write table
output_name = ['week2_v4_accCheck_toe',num2str(data.toe(1)),'.csv'];
if isfile(output_name)
    disp(['Outputfile ', output_name ,' already exists. Remove it first if you wish to generate a new one.']);
else
    writetable(data,output_name);
    disp(['Data table written to file ', output_name, '.']);
end
