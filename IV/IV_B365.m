% --------------------------------------------------------------------------
%
% This script reads the database file and generates the Train and Test sets,
% using B365 data. Then it performs KMeans classifications, to predict the match results.
%
% -------------------------------------------------------------------------

%Initialize workspace.
clc
clear

% Define the database file.
database_file = '../database.sqlite';

% Create a connection to the database.
fprintf('Establishing connection to %s\n',database_file);
conn = sqlite(database_file,'readonly');

% Retrieval of matches for B365. Records with empty B365H, B365D and B365A are ignored.
B365ColumnNames = {'B365H','B365D','B365A'};
B365Query = 'select B365H, B365D, B365A from Match where B365H != 0 and B365D != 0 and B365A !=0 ';
% Retriaval of matches that Home team won.
B365HQuery= [B365Query,'and home_team_goal>away_team_goal'];
B365HCell = fetch(conn,B365HQuery);
B365H = cell2table(B365HCell,'VariableNames',B365ColumnNames);
clear B365HCell
% Retriaval of matches that resulted in a draw.
B365DQuery= [B365Query,'and home_team_goal=away_team_goal'];
B365DCell = fetch(conn,B365DQuery);
B365D = cell2table(B365DCell,'VariableNames',B365ColumnNames);
clear B365DCell
% Retriaval of matches that Away team won.
B365AQuery= [B365Query,'and home_team_goal<away_team_goal'];
B365ACell = fetch(conn,B365AQuery);
B365A = cell2table(B365ACell,'VariableNames',B365ColumnNames);
clear B365ACell

% Close connection.
close(conn);
fprintf('Closed connection to database: %s\n',database_file);

N = 1000;
T = [ones(1,N),2*ones(1,N),3*ones(1,N)];
T = T';

% B365 Kmeans wich C=3.
B365HKmeans = table2array(B365H(1:N,:));
B365DKmeans = table2array(B365D(1:N,:));
B365AKmeans = table2array(B365A(1:N,:));
figure(1)
plot3(B365HKmeans(:,1),B365HKmeans(:,2),B365HKmeans(:,3),'r.','MarkerSize',12)
hold on
plot3(B365DKmeans(:,1),B365DKmeans(:,2),B365DKmeans(:,3),'g.','MarkerSize',12)
hold on
plot3(B365AKmeans(:,1),B365AKmeans(:,2),B365AKmeans(:,3),'b.','MarkerSize',12)
title('B365 Original Points')
legend('H','D','A')
xlabel('B365H')
ylabel('B365D')
zlabel('B365A')

B365 = [double(B365HKmeans);double(B365DKmeans);double(B365AKmeans)];
B365idx = kmeans(B365,3);
figure(2)
plot3(B365(B365idx==1,1),B365(B365idx==1,2),B365(B365idx==1,3),'r.','MarkerSize',12)
hold on
plot3(B365(B365idx==2,1),B365(B365idx==2,2),B365(B365idx==2,3),'g.','MarkerSize',12)
hold on
plot3(B365(B365idx==3,1),B365(B365idx==3,2),B365(B365idx==3,3),'b.','MarkerSize',12)
title('B365 KMeans Points')
legend('H','D','A')
xlabel('B365H')
ylabel('B365D')
zlabel('B365A')

B365KMeansCorrectClassificationRatio = 1 - (sum(B365idx~=T) / (3*N));
fprintf('Kmeans correct Classification Ratio for B365: %0.4f.\n',B365KMeansCorrectClassificationRatio);

B365C1HD=0;
B365C1DD=0;
B365C1AD=0;
B365C2HD=0;
B365C2DD=0;
B365C2AD=0;
B365C3HD=0;
B365C3DD=0;
B365C3AD=0;
for i = 1:(N-1)
    if B365idx(i) == 1 
        B365C1HD = B365C1HD + 1;
    elseif B365idx(i) == 2 
        B365C2HD = B365C2HD + 1;
    elseif B365idx(i) == 3
        B365C3HD = B365C3HD + 1;
    end
end
for i = N:(2*N-1)
    if B365idx(i) == 1 
        B365C1DD = B365C1DD + 1;
    elseif B365idx(i) == 2 
        B365C2DD = B365C2DD + 1;
    elseif B365idx(i) == 3
        B365C3DD = B365C3DD + 1;
    end
end
for i = 2*N:3*N
    if B365idx(i) == 1 
        B365C1AD = B365C1AD + 1;
    elseif B365idx(i) == 2 
        B365C2AD = B365C2AD + 1;
    elseif B365idx(i) == 3
        B365C3AD = B365C3AD + 1;
    end
end

figure(3)
X = categorical({'H','D','A'});
X = reordercats(X,{'H','D','A'});
Y = [B365C1HD, B365C1DD, B365C1AD];
bar(X,Y)
title('B365 CLuster 1 Density')

maxY = max(Y);
if B365C1HD == maxY
    disp('Home wins are dominating Cluster 1.')
elseif B365C1DD == maxY
   disp('Draws are dominating Cluster 1.')
elseif B365C1AD == maxY
    disp('Away wins are dominating Cluster 1.')
end

figure(4)
X = categorical({'H','D','A'});
X = reordercats(X,{'H','D','A'});
Y = [B365C2HD, B365C2DD, B365C2AD];
bar(X,Y)
title('B365 CLuster 2 Density')

maxY = max(Y);
if B365C2HD == maxY
    disp('Home wins are dominating Cluster 2.')
elseif B365C2DD == maxY
   disp('Draws are dominating Cluster 2.')
elseif B365C2AD == maxY
    disp('Away wins are dominating Cluster 2.')
end

figure(5)
X = categorical({'H','D','A'});
X = reordercats(X,{'H','D','A'});
Y = [B365C3HD, B365C3DD, B365C3AD];
bar(X,Y)
title('B365 CLuster 3 Density')

maxY = max(Y);
if B365C3HD == maxY
    disp('Home wins are dominating Cluster 3.')
elseif B365C3DD == maxY
   disp('Draws are dominating Cluster 3.')
elseif B365C3AD == maxY
    disp('Away wins are dominating Cluster 3.')
end
