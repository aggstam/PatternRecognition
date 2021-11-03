%Initialize workspace.
clc
clear

% Define the database file.
database_file = '../database.sqlite';

% Create a connection to the database.
fprintf('Establishing connection to %s\n',database_file);
conn = sqlite(database_file,'readonly');

% Retrieval of matches for BW. Records with empty BWH, BWD and BWA are ignored.
BWColumnNames = {'BWH','BWD','BWA'};
BWQuery = 'select BWH, BWD, BWA from Match where BWH != 0 and BWD != 0 and BWA !=0 ';
% Retriaval of matches that Home team won.
BWHQuery= [BWQuery,'and home_team_goal>away_team_goal'];
BWHCell = fetch(conn,BWHQuery);
BWH = cell2table(BWHCell,'VariableNames',BWColumnNames);
clear BWHCell
% Retriaval of matches that resulted in a draw.
BWDQuery= [BWQuery,'and home_team_goal=away_team_goal'];
BWDCell = fetch(conn,BWDQuery);
BWD = cell2table(BWDCell,'VariableNames',BWColumnNames);
clear BWDCell
% Retriaval of matches that Away team won.
BWAQuery= [BWQuery,'and home_team_goal<away_team_goal'];
BWACell = fetch(conn,BWAQuery);
BWA = cell2table(BWACell,'VariableNames',BWColumnNames);
clear BWACell

% Close connection.
close(conn);
fprintf('Closed connection to database: %s\n',database_file);

N = 1000;
T = [ones(1,N),2*ones(1,N),3*ones(1,N)];
T = T';

% BW Kmeans wich C=3.
BWHKmeans = table2array(BWH(1:N,:));
BWDKmeans = table2array(BWD(1:N,:));
BWAKmeans = table2array(BWA(1:N,:));
figure(1)
plot3(BWHKmeans(:,1),BWHKmeans(:,2),BWHKmeans(:,3),'r.','MarkerSize',12)
hold on
plot3(BWDKmeans(:,1),BWDKmeans(:,2),BWDKmeans(:,3),'g.','MarkerSize',12)
hold on
plot3(BWAKmeans(:,1),BWAKmeans(:,2),BWAKmeans(:,3),'b.','MarkerSize',12)
title('BW Original Points')
legend('H','D','A')
xlabel('BWH')
ylabel('BWD')
zlabel('BWA')

BW = [double(BWHKmeans);double(BWDKmeans);double(BWAKmeans)];
BWidx = kmeans(BW,3);
figure(2)
plot3(BW(BWidx==1,1),BW(BWidx==1,2),BW(BWidx==1,3),'r.','MarkerSize',12)
hold on
plot3(BW(BWidx==2,1),BW(BWidx==2,2),BW(BWidx==2,3),'g.','MarkerSize',12)
hold on
plot3(BW(BWidx==3,1),BW(BWidx==3,2),BW(BWidx==3,3),'b.','MarkerSize',12)
title('BW KMeans Points')
legend('H','D','A')
xlabel('BWH')
ylabel('BWD')
zlabel('BWA')

BWKMeansCorrectClassificationRatio = 1 - (sum(BWidx~=T) / (3*N));
fprintf('Kmeans correct Classification Ratio for BW: %0.4f.\n',BWKMeansCorrectClassificationRatio);

BWC1HD=0;
BWC1DD=0;
BWC1AD=0;
BWC2HD=0;
BWC2DD=0;
BWC2AD=0;
BWC3HD=0;
BWC3DD=0;
BWC3AD=0;
for i = 1:(N-1)
    if BWidx(i) == 1 
        BWC1HD = BWC1HD + 1;
    elseif BWidx(i) == 2 
        BWC2HD = BWC2HD + 1;
    elseif BWidx(i) == 3
        BWC3HD = BWC3HD + 1;
    end
end
for i = N:(2*N-1)
    if BWidx(i) == 1 
        BWC1DD = BWC1DD + 1;
    elseif BWidx(i) == 2 
        BWC2DD = BWC2DD + 1;
    elseif BWidx(i) == 3
        BWC3DD = BWC3DD + 1;
    end
end
for i = 2*N:3*N
    if BWidx(i) == 1 
        BWC1AD = BWC1AD + 1;
    elseif BWidx(i) == 2 
        BWC2AD = BWC2AD + 1;
    elseif BWidx(i) == 3
        BWC3AD = BWC3AD + 1;
    end
end

figure(3)
X = categorical({'H','D','A'});
X = reordercats(X,{'H','D','A'});
Y = [BWC1HD, BWC1DD, BWC1AD];
bar(X,Y)
title('BW CLuster 1 Density')

maxY = max(Y);
if BWC1HD == maxY
    disp('Home wins are dominating Cluster 1.')
elseif BWC1DD == maxY
   disp('Draws are dominating Cluster 1.')
elseif BWC1AD == maxY
    disp('Away wins are dominating Cluster 1.')
end

figure(4)
X = categorical({'H','D','A'});
X = reordercats(X,{'H','D','A'});
Y = [BWC2HD, BWC2DD, BWC2AD];
bar(X,Y)
title('BW CLuster 2 Density')

maxY = max(Y);
if BWC2HD == maxY
    disp('Home wins are dominating Cluster 2.')
elseif BWC2DD == maxY
   disp('Draws are dominating Cluster 2.')
elseif BWC2AD == maxY
    disp('Away wins are dominating Cluster 2.')
end

figure(5)
X = categorical({'H','D','A'});
X = reordercats(X,{'H','D','A'});
Y = [BWC3HD, BWC3DD, BWC3AD];
bar(X,Y)
title('BW CLuster 3 Density')

maxY = max(Y);
if BWC3HD == maxY
    disp('Home wins are dominating Cluster 3.')
elseif BWC3DD == maxY
   disp('Draws are dominating Cluster 3.')
elseif BWC3AD == maxY
    disp('Away wins are dominating Cluster 3.')
end