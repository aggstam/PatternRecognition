% --------------------------------------------------------------------------
%
% This script reads the database file and generates the Train and Test sets,
% using IW data. Then it performs KMeans classifications, to predict the match results.
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

% Retrieval of matches for IW. Records with empty IWH, IWD and IWA are ignored.
IWColumnNames = {'IWH','IWD','IWA'};
IWQuery = 'select IWH, IWD, IWA from Match where IWH != 0 and IWD != 0 and IWA !=0 ';
% Retriaval of matches that Home team won.
IWHQuery= [IWQuery,'and home_team_goal>away_team_goal'];
IWHCell = fetch(conn,IWHQuery);
IWH = cell2table(IWHCell,'VariableNames',IWColumnNames);
clear IWHCell
% Retriaval of matches that resulted in a draw.
IWDQuery= [IWQuery,'and home_team_goal=away_team_goal'];
IWDCell = fetch(conn,IWDQuery);
IWD = cell2table(IWDCell,'VariableNames',IWColumnNames);
clear IWDCell
% Retriaval of matches that Away team won.
IWAQuery= [IWQuery,'and home_team_goal<away_team_goal'];
IWACell = fetch(conn,IWAQuery);
IWA = cell2table(IWACell,'VariableNames',IWColumnNames);
clear IWACell

% Close connection.
close(conn);
fprintf('Closed connection to database: %s\n',database_file);

N = 1000;
T = [ones(1,N),2*ones(1,N),3*ones(1,N)];
T = T';

% IW Kmeans wich C=3.
IWHKmeans = table2array(IWH(1:N,:));
IWDKmeans = table2array(IWD(1:N,:));
IWAKmeans = table2array(IWA(1:N,:));
figure(1)
plot3(IWHKmeans(:,1),IWHKmeans(:,2),IWHKmeans(:,3),'r.','MarkerSize',12)
hold on
plot3(IWDKmeans(:,1),IWDKmeans(:,2),IWDKmeans(:,3),'g.','MarkerSize',12)
hold on
plot3(IWAKmeans(:,1),IWAKmeans(:,2),IWAKmeans(:,3),'b.','MarkerSize',12)
title('IW Original Points')
legend('H','D','A')
xlabel('IWH')
ylabel('IWD')
zlabel('IWA')

IW = [double(IWHKmeans);double(IWDKmeans);double(IWAKmeans)];
IWidx = kmeans(IW,3);
figure(2)
plot3(IW(IWidx==1,1),IW(IWidx==1,2),IW(IWidx==1,3),'r.','MarkerSize',12)
hold on
plot3(IW(IWidx==2,1),IW(IWidx==2,2),IW(IWidx==2,3),'g.','MarkerSize',12)
hold on
plot3(IW(IWidx==3,1),IW(IWidx==3,2),IW(IWidx==3,3),'b.','MarkerSize',12)
title('IW KMeans Points')
legend('H','D','A')
xlabel('IWH')
ylabel('IWD')
zlabel('IWA')

IWKMeansCorrectClassificationRatio = 1 - (sum(IWidx~=T) / (3*N));
fprintf('Kmeans correct Classification Ratio for IW: %0.4f.\n',IWKMeansCorrectClassificationRatio);

IWC1HD=0;
IWC1DD=0;
IWC1AD=0;
IWC2HD=0;
IWC2DD=0;
IWC2AD=0;
IWC3HD=0;
IWC3DD=0;
IWC3AD=0;
for i = 1:(N-1)
    if IWidx(i) == 1 
        IWC1HD = IWC1HD + 1;
    elseif IWidx(i) == 2 
        IWC2HD = IWC2HD + 1;
    elseif IWidx(i) == 3
        IWC3HD = IWC3HD + 1;
    end
end
for i = N:(2*N-1)
    if IWidx(i) == 1 
        IWC1DD = IWC1DD + 1;
    elseif IWidx(i) == 2 
        IWC2DD = IWC2DD + 1;
    elseif IWidx(i) == 3
        IWC3DD = IWC3DD + 1;
    end
end
for i = 2*N:3*N
    if IWidx(i) == 1 
        IWC1AD = IWC1AD + 1;
    elseif IWidx(i) == 2 
        IWC2AD = IWC2AD + 1;
    elseif IWidx(i) == 3
        IWC3AD = IWC3AD + 1;
    end
end

figure(3)
X = categorical({'H','D','A'});
X = reordercats(X,{'H','D','A'});
Y = [IWC1HD, IWC1DD, IWC1AD];
bar(X,Y)
title('IW CLuster 1 Density')

maxY = max(Y);
if IWC1HD == maxY
    disp('Home wins are dominating Cluster 1.')
elseif IWC1DD == maxY
   disp('Draws are dominating Cluster 1.')
elseif IWC1AD == maxY
    disp('Away wins are dominating Cluster 1.')
end

figure(4)
X = categorical({'H','D','A'});
X = reordercats(X,{'H','D','A'});
Y = [IWC2HD, IWC2DD, IWC2AD];
bar(X,Y)
title('IW CLuster 2 Density')

maxY = max(Y);
if IWC2HD == maxY
    disp('Home wins are dominating Cluster 2.')
elseif IWC2DD == maxY
   disp('Draws are dominating Cluster 2.')
elseif IWC2AD == maxY
    disp('Away wins are dominating Cluster 2.')
end

figure(5)
X = categorical({'H','D','A'});
X = reordercats(X,{'H','D','A'});
Y = [IWC3HD, IWC3DD, IWC3AD];
bar(X,Y)
title('IW CLuster 3 Density')

maxY = max(Y);
if IWC3HD == maxY
    disp('Home wins are dominating Cluster 3.')
elseif IWC3DD == maxY
   disp('Draws are dominating Cluster 3.')
elseif IWC3AD == maxY
    disp('Away wins are dominating Cluster 3.')
end
