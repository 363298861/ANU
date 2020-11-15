%%
% Assignment-1: Hough Transform
% Your name (Your uniID)
%
filename = 'source.jpg';
% Load the image
img = imread(filename);
img = rgb2gray(img);

% Edge detection
edge_img = edge(img, 'canny');
imwrite(edge_img, strcat('edges-detected-',filename));

% Detect lines
[accumulator, rhos, thetas] = hough_line(edge_img, 1);

% Get the line with maximum votes (longest line)
[rho, theta] = peak_votes(accumulator, rhos, thetas);

% Draw lines in the image
figure, imshow(img), hold on
draw(rho, theta);

saveas(gcf, strcat('lines-detected-',filename));

%% Hough transform for lines
function [accumulator, rhos, thetas] = hough_line(img, theta_step)
% Input:
% img - 2D binary image with nonzeros representing edges
% theta_step - Spacing between angles 
%               between 0 and 360 degrees. Default step is 1.
% Returns:
% accumulator - 2D array of the hough transform accumulator
% rhos - array of rho values. Max size is 2 times the diagonal
%        distance of the input image. [-diag_len, diag_len]
% theta - array of angles used in computation, in radians. [-pi/2, pi/2]

    % rho and theta ranges
    thetas = deg2rad(-90:theta_step:90);
    [width, height] = size(img);
    diag_len = round(norm([width, height]));
    rhos = (-diag_len:1:diag_len);

    % Cache some resuable values
    cos_t = cos(thetas);
    sin_t = sin(thetas);
    num_thetas = length(thetas);
    num_rhos = length(rhos);

    % Hough accumulator array of theta vs rho
    accumulator = zeros(num_rhos, num_thetas);
    % (row, col) indexes to edges
    [y_idxs, x_idxs] = find(img);

    % Vote in the hough accumulator
    for i = 1 : length(x_idxs)
        x = x_idxs(i);
        y = y_idxs(i);

        for t_idx = 1 : num_thetas
            % Calculate rho and add diag_len to it to make it a positive index
            r = round(x * cos_t(t_idx) + y * sin_t(t_idx));
            ir = r + diag_len + 1;
            accumulator(ir, t_idx) = accumulator(ir, t_idx) + 1;
        end
    end
end

%% Finds the index with max number of votes in the hough accumulator
function [rho, theta] = peak_votes(accumulator, rhos, thetas)
    [x,y] = find(accumulator == max(accumulator(:)));
    rho = rhos(x);
    theta = thetas(y);
end

%% draws line in the image
function draw(rho, theta)
    a = cos(theta);
    b = sin(theta);
    x0 = a * rho;
    y0 = b * rho;
    pt1 = [round(x0 + 1000*(-b)), round(y0 + 1000*(a))];
    pt2 = [round(x0 - 1000*(-b)), round(y0 - 1000*(a))];
    line([pt1(1),pt2(1)], [pt1(2),pt2(2)], 'LineWidth', 2, 'Color', 'red');
end






