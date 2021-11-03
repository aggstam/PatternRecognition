%Initialize workspace.
clc
clear

% Define the database file.
database_file = '../database.sqlite';

% Create a connection to the database.
fprintf('Establishing connection to %s\n',database_file);
conn = sqlite(database_file,'readonly');

TeamAttributesColumnNames = {'home_buildUpPlaySpeed','home_buildUpPlayPassing','home_chanceCreationPassing','home_chanceCreationCrossing',...
'home_chanceCreationShooting','home_defencePressure','home_defenceAggression','home_defenceTeamWidth',...
'away_buildUpPlaySpeed','away_buildUpPlayPassing','away_chanceCreationPassing','away_chanceCreationCrossing',...
'away_chanceCreationShooting','away_defencePressure','away_defenceAggression','away_defenceTeamWidth',...
'B365H','B365D','B365A','BWH','BWD','BWA','IWH','IWD','IWA','LBH','LBD','LBA'};
QueryTeamAttributesColumns = ['select h.buildUpPlaySpeed, h.buildUpPlayPassing, h.chanceCreationPassing, h.chanceCreationCrossing, h.chanceCreationShooting, h.defencePressure, h.defenceAggression, h.defenceTeamWidth, '...
                              'a.buildUpPlaySpeed, a.buildUpPlayPassing, a.chanceCreationPassing, a.chanceCreationCrossing, a.chanceCreationShooting, a.defencePressure, a.defenceAggression, a.defenceTeamWidth, '...
                              'm.B365H, m.B365D, m.B365A, m.BWH, m.BWD, m.BWA, m.IWH, m.IWD, m.IWA, m.LBH, m.LBD, m.LBA '...
                              'from Match m inner join Team_Attributes h on m.home_team_api_id=h.team_api_id inner join Team_Attributes a on m.away_team_api_id=a.team_api_id '...
                              'where strftime(''%Y'',m.date)=strftime(''%Y'',h.date) and strftime(''%Y'',m.date)=strftime(''%Y'',a.date) '...
                              'and m.B365H != 0 and m.B365D != 0 and m.B365A !=0 '...
                              'and m.BWH != 0 and m.BWD != 0 and m.BWA !=0 '...
                              'and m.IWH != 0 and m.IWD != 0 and m.IWA !=0 '...
                              'and m.LBH != 0 and m.LBD != 0 and m.LBA !=0 '];

HQuery= [QueryTeamAttributesColumns,'and m.home_team_goal>m.away_team_goal'];
HCell = fetch(conn,HQuery);
H = cell2table(HCell,'VariableNames',TeamAttributesColumnNames);
clear HCell
% Retriaval of matches that resulted in a draw.
DQuery= [QueryTeamAttributesColumns,'and m.home_team_goal=m.away_team_goal'];
DCell = fetch(conn,DQuery);
D = cell2table(DCell,'VariableNames',TeamAttributesColumnNames);
clear DCell
% Retriaval of matches that Away team won.
AQuery= [QueryTeamAttributesColumns,'and m.home_team_goal<m.away_team_goal'];
ACell = fetch(conn,AQuery);
A = cell2table(ACell,'VariableNames',TeamAttributesColumnNames);
clear ACell

% Close connection.
close(conn);
fprintf('Closed connection to database: %s\n',database_file);

N = 1000; % Number of records from each H, D and A tables. 3000 total records.
K = 10; % Folders
M = N / K; % Testing set records. Train and Test sets will contain 2700 and 300 records respectively, for each folder.
Rows = [1:N]; % Table containing each row index.

% Initializing H, D and A tables of each folder.
HElements = [double(table2array(H(1:N,:)))];
TrainHElements = cell(1,K);
TestHElements = cell(1,K);
DElements = [double(table2array(D(1:N,:)))];
TrainDElements = cell(1,K);
TestDElements = cell(1,K);
AElements = [double(table2array(A(1:N,:)))];
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
    net = newff(PTrain,TTrain,[14 7 1],{'tansig' 'tansig' 'purelin'});
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
fprintf('Correct Classication Ratio median(10-Fold Validation, multi-layer neural network) for Train sample: %0.4f.\n',TrainCorrectClassificationRatio);
TestCorrectClassificationRatio = TestCorrectClassificationRatioSum / K;
fprintf('Correct Classication Ratio median(10-Fold Validation, multiy-layer neural network) for Test sample: %0.4f.\n',TestCorrectClassificationRatio);
