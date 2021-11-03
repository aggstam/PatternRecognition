function [TrainCorrectClassificationRatio, TestCorrectClassificationRatio] = II_B365(conn)

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

N = 1000; % Number of records from each H, D and A tables. 3000 total records.
K = 10; % Folders
M = N / K; % Testing set records. Train and Test sets will contain 2700 and 300 records respectively, for each folder.
Rows = [1:N]; % Table containing each row index.

% Initializing H, D and A tables of each folder.
HElements = [double(table2array(B365H(1:N,:)))];
TrainHElements = cell(1,K);
TestHElements = cell(1,K);
DElements = [double(table2array(B365D(1:N,:)))];
TrainDElements = cell(1,K);
TestDElements = cell(1,K);
AElements = [double(table2array(B365A(1:N,:)))];
TrainAElements = cell(1,K);
TestAElements = cell(1,K);
% Extracting TestElements and TrainElements tables for H, A and D records respectively.
% Tables structure:
%   1. TestElements: 10x{100x28}
%   2. TrainElements: 10x{900x28}
for k = 1:1:K
    % We move 100 records in each loop to aquire Test indexes.
    TestRows = Rows(((k-1)*M)+1:k*M);
    % Rest 900 indexes will be the Train set.
    TrainRows = setdiff(Rows,TestRows);
    
    % Extraction of Train and Test folder sets, based on index tables
    % above.
    TestHElements{k} = HElements(TestRows,:);
    TrainHElements{k} = HElements(TrainRows,:);
    TestDElements{k} = DElements(TestRows,:);
    TrainDElements{k} = DElements(TrainRows,:);
    TestAElements{k} = AElements(TestRows,:);
    TrainAElements{k} = AElements(TrainRows,:);
end;

% Algorith will be executed K(10) times, computing using table sets of each
% folder created previously.
NTrain = 900;
NTest = 100;

% We use the result sum of each set, for median computation.
TrainCorrectClassificationRatioSum = 0;
TestCorrectClassificationRatioSum = 0;
for k = 1:1:K
    % Generate training patterns from classes.
    HTrain = TrainHElements{k};
    DTrain = TrainDElements{k};
    ATrain = TrainAElements{k};
    
    % Set the training patterns matrix for the feed forward neural network object.
    PTrain = [HTrain;DTrain;ATrain];
    PTrain = PTrain';
    % Set the target vector corresponding to the training patterns stored in P.
    TTrain = [ones(1,NTrain),2*ones(1,NTrain),3*ones(1,NTrain)];
    
    % Set the neural network
    net = newff(PTrain,TTrain,[3 3 2 1],{'tansig' 'tansig' 'tansig' 'purelin'});
    init(net);
    net.trainParam.epochs = 500;
    net.trainParam.showCommandLine = 0;
    net.trainParam.goal = 0.0;
    net = train(net,PTrain,TTrain);
    
    % Check network performance on training patterns.
    EstimatedTrainingTargets = sim(net,PTrain);
    EstimatedTrainingTargets = round(EstimatedTrainingTargets);
    TrainCorrectClassificationRatioSum = TrainCorrectClassificationRatioSum + (1 - (sum(EstimatedTrainingTargets~=TTrain) / (3*NTrain)));
    
    % Generate testing patterns from classes.
    HTest = TestHElements{k};
    DTest = TestDElements{k};
    ATest = TestAElements{k};
    % Check network performance on test patterns.
    PTest = [HTest;DTest;ATest];
    PTest = PTest';
    TTest = [ones(1,NTest),2*ones(1,NTest),3*ones(1,NTest)];
    EstimatedTestTargets = sim(net,PTest);
    EstimatedTestTargets = round(EstimatedTestTargets);
    TestCorrectClassificationRatioSum = TestCorrectClassificationRatioSum + (1 - (sum(EstimatedTestTargets~=TTest) / (3*NTest)));
end;

% K-fold(10-fold) validation median computation.
TrainCorrectClassificationRatio = TrainCorrectClassificationRatioSum / K;
fprintf('B365 Correct Classication Ratio median(10-Fold Validation, multi-layer neural network) for Train sample: %0.4f.\n',TrainCorrectClassificationRatio);
TestCorrectClassificationRatio = TestCorrectClassificationRatioSum / K;
fprintf('B365 Correct Classication Ratio median(10-Fold Validation, multi-layer neural network) for Test sample: %0.4f.\n',TestCorrectClassificationRatio);

end