% Define the data
data = [
    3, 100, 0.8261318463963382;
    3, 300, 0.8129217356455434;
    3, 500, 0.8306118851609866;
    5, 100, 0.8392828287065404;
    5, 300, 0.8433716707021792;
    5, 500, 0.8478464651527237;
    7, 100, 0.8401819620075988;
    7, 300, 0.8390375749972253;
    7, 500, 0.8303322952133313;
    9, 100, 0.839010553569853;
    9, 300, 0.8006858396141656;
    9, 500, 0.8364057490354369;
    11, 100, 0.7870075848170737;
    11, 300, 0.8087108942856609;
    11, 500, 0.8485272840487242;
];

% Extract the unique values for kernel size, epochs, and MCC
kernel_sizes = unique(data(:, 1));
epochs = unique(data(:, 2));
mcc = data(:, 3);

% Create a 3D bar plot
figure;
h = bar3(reshape(mcc, length(epochs), length(kernel_sizes)));

% Set labels and titles
xlabel('Kernel Size');
xticklabels(cellstr(num2str(kernel_sizes)));
ylabel('Epochs');
yticklabels(cellstr(num2str(epochs)));
zlabel('MCC');
title('3D Bar Plot of MCC');

% Adjust the view for better visualization
view(-30, 30);

% Set the Z-axis limits from 0.7 to 1
zlim([0.7, 1]);

% Customize the colormap to 'cool'
colormap("hsv");

% Add color bar
colorbar;

% Define a colormap for the bars
colors = colormap("hsv");

% Map colors to MCC values
cdata = (mcc - min(mcc)) / (max(mcc) - min(mcc)) * (size(colors, 1) - 1) + 1;

% Iterate through the bars and set the FaceColor
for k = 1:length(h)
    h(k).CData = cdata(k);
    h(k).FaceColor = colors(round(cdata(k)), :);
end