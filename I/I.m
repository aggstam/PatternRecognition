% --------------------------------------------------------------------------
%
% This script reads the database file and generates the Train and Test
% sets for each company.
%
% --------------------------------------------------------------------------

%Initialize workspace.
clc
clear

% Define the database file.
database_file = '../database.sqlite';

% Create a connection to the database.
fprintf('Establishing connection to %s\n',database_file);
conn = sqlite(database_file,'readonly');

[B365TrainCorrectClassificationRatio, B365TestCorrectClassificationRatio] = I_B365(conn);
[BWTrainCorrectClassificationRatio, BWTestCorrectClassificationRatio] = I_BW(conn);
[IWTrainCorrectClassificationRatio, IWTestCorrectClassificationRatio] = I_IW(conn);
[LBTrainCorrectClassificationRatio, LBTestCorrectClassificationRatio] = I_LB(conn);

% Close connection.
close(conn);
fprintf('Closed connection to database: %s\n',database_file);

TrainClassificationRations = [B365TrainCorrectClassificationRatio, BWTrainCorrectClassificationRatio, IWTrainCorrectClassificationRatio, LBTrainCorrectClassificationRatio];
maxTrainClassificationRation = max(TrainClassificationRations);
if B365TrainCorrectClassificationRatio == maxTrainClassificationRation
    fprintf('B365 has the best Classication Ratio(%0.4f) for Train sample, using 10-Fold Validation on a linear neural network.\n',maxTrainClassificationRation);
elseif BWTrainCorrectClassificationRatio == maxTrainClassificationRation
    fprintf('BW has the best Classication Ratio(%0.4f) for Train sample, using 10-Fold Validation on a linear neural network.\n',maxTrainClassificationRation);
elseif IWTrainCorrectClassificationRatio == maxTrainClassificationRation
    fprintf('IW has the best Classication Ratio(%0.4f) for Train sample, using 10-Fold Validation on a linear neural network.\n',maxTrainClassificationRation);
elseif LBTrainCorrectClassificationRatio == maxTrainClassificationRation
    fprintf('LB has the best Classication Ratio(%0.4f) for Train sample, using 10-Fold Validation on a linear neural network.\n',maxTrainClassificationRation);
end

TestClassificationRations = [B365TestCorrectClassificationRatio, BWTestCorrectClassificationRatio, IWTestCorrectClassificationRatio, LBTestCorrectClassificationRatio];
maxTestClassificationRations = max(TestClassificationRations);
if B365TestCorrectClassificationRatio == maxTestClassificationRations
    fprintf('B365 has the best Classication Ratio(%0.4f) for Test sample, using 10-Fold Validation on a linear neural network.\n',maxTestClassificationRations);
elseif BWTestCorrectClassificationRatio == maxTestClassificationRations
    fprintf('BW has the best Classication Ratio(%0.4f) for Test sample, using 10-Fold Validation on a linear neural network.\n',maxTestClassificationRations);
elseif IWTestCorrectClassificationRatio == maxTestClassificationRations
    fprintf('IW has the best Classication Ratio(%0.4f) for Test sample, using 10-Fold Validation on a linear neural network.\n',maxTestClassificationRations);
elseif LBTestCorrectClassificationRatio == LBTestCorrectClassificationRatio
    fprintf('LB has the best Classication Ratio(%0.4f) for Test sample, using 10-Fold Validation on a linear neural network.\n',maxTestClassificationRations);
end
