% --------------------------------------------------------------------------
%
% This script reads the database file and generates the Train and Test sets,
% using LB data. Then it performs KMeans classifications, to predict the match results.
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

% Retrieval of matches for LB. Records with empty LBH, LBD and LBA are ignored.
LBColumnNames = {'LBH','LBD','LBA'};
LBQuery = 'select LBH, LBD, LBA from Match where LBH != 0 and LBD != 0 and LBA !=0 ';
% Retriaval of matches that Home team won.
LBHQuery= [LBQuery,'and home_team_goal>away_team_goal'];
LBHCell = fetch(conn,LBHQuery);
LBH = cell2table(LBHCell,'VariableNames',LBColumnNames);
clear LBHCell
% Retriaval of matches that resulted in a draw.
LBDQuery= [LBQuery,'and home_team_goal=away_team_goal'];
LBDCell = fetch(conn,LBDQuery);
LBD = cell2table(LBDCell,'VariableNames',LBColumnNames);
clear LBDCell
% Retriaval of matches that Away team won.
LBAQuery= [LBQuery,'and home_team_goal<away_team_goal'];
LBACell = fetch(conn,LBAQuery);
LBA = cell2table(LBACell,'VariableNames',LBColumnNames);
clear LBACell

% Close connection.
close(conn);
fprintf('Closed connection to database: %s\n',database_file);

N = 1000;
T = [ones(1,N),2*ones(1,N),3*ones(1,N)];
T = T';

% LB Kmeans wich C=3.
LBHKmeans = table2array(LBH(1:N,:));
LBDKmeans = table2array(LBD(1:N,:));
LBAKmeans = table2array(LBA(1:N,:));
figure(1)
plot3(LBHKmeans(:,1),LBHKmeans(:,2),LBHKmeans(:,3),'r.','MarkerSize',12)
hold on
plot3(LBDKmeans(:,1),LBDKmeans(:,2),LBDKmeans(:,3),'g.','MarkerSize',12)
hold on
plot3(LBAKmeans(:,1),LBAKmeans(:,2),LBAKmeans(:,3),'b.','MarkerSize',12)
title('LB Original Points')
legend('H','D','A')
xlabel('LBH')
ylabel('LBD')
zlabel('LBA')

LB = [double(LBHKmeans);double(LBDKmeans);double(LBAKmeans)];
LBidx = kmeans(LB,3);
figure(2)
plot3(LB(LBidx==1,1),LB(LBidx==1,2),LB(LBidx==1,3),'r.','MarkerSize',12)
hold on
plot3(LB(LBidx==2,1),LB(LBidx==2,2),LB(LBidx==2,3),'g.','MarkerSize',12)
hold on
plot3(LB(LBidx==3,1),LB(LBidx==3,2),LB(LBidx==3,3),'b.','MarkerSize',12)
title('LB KMeans Points')
legend('H','D','A')
xlabel('LBH')
ylabel('LBD')
zlabel('LBA')

LBKMeansCorrectClassificationRatio = 1 - (sum(LBidx~=T) / (3*N));
fprintf('Kmeans correct Classification Ratio for LB: %0.4f.\n',LBKMeansCorrectClassificationRatio);

LBC1HD=0;
LBC1DD=0;
LBC1AD=0;
LBC2HD=0;
LBC2DD=0;
LBC2AD=0;
LBC3HD=0;
LBC3DD=0;
LBC3AD=0;
for i = 1:(N-1)
    if LBidx(i) == 1 
        LBC1HD = LBC1HD + 1;
    elseif LBidx(i) == 2 
        LBC2HD = LBC2HD + 1;
    elseif LBidx(i) == 3
        LBC3HD = LBC3HD + 1;
    end
end
for i = N:(2*N-1)
    if LBidx(i) == 1 
        LBC1DD = LBC1DD + 1;
    elseif LBidx(i) == 2 
        LBC2DD = LBC2DD + 1;
    elseif LBidx(i) == 3
        LBC3DD = LBC3DD + 1;
    end
end
for i = 2*N:3*N
    if LBidx(i) == 1 
        LBC1AD = LBC1AD + 1;
    elseif LBidx(i) == 2 
        LBC2AD = LBC2AD + 1;
    elseif LBidx(i) == 3
        LBC3AD = LBC3AD + 1;
    end
end

figure(3)
X = categorical({'H','D','A'});
X = reordercats(X,{'H','D','A'});
Y = [LBC1HD, LBC1DD, LBC1AD];
bar(X,Y)
title('LB CLuster 1 Density')

maxY = max(Y);
if LBC1HD == maxY
    disp('Home wins are dominating Cluster 1.')
elseif LBC1DD == maxY
   disp('Draws are dominating Cluster 1.')
elseif LBC1AD == maxY
    disp('Away wins are dominating Cluster 1.')
end

figure(4)
X = categorical({'H','D','A'});
X = reordercats(X,{'H','D','A'});
Y = [LBC2HD, LBC2DD, LBC2AD];
bar(X,Y)
title('LB CLuster 2 Density')

maxY = max(Y);
if LBC2HD == maxY
    disp('Home wins are dominating Cluster 2.')
elseif LBC2DD == maxY
   disp('Draws are dominating Cluster 2.')
elseif LBC2AD == maxY
    disp('Away wins are dominating Cluster 2.')
end

figure(5)
X = categorical({'H','D','A'});
X = reordercats(X,{'H','D','A'});
Y = [LBC3HD, LBC3DD, LBC3AD];
bar(X,Y)
title('LB CLuster 3 Density')

maxY = max(Y);
if LBC3HD == maxY
    disp('Home wins are dominating Cluster 3.')
elseif LBC3DD == maxY
   disp('Draws are dominating Cluster 3.')
elseif LBC3AD == maxY
    disp('Away wins are dominating Cluster 3.')
end
