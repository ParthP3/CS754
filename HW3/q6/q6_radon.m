image = imread('cryoem.png');
%NN = [50 100 500 1000 2000 5000 10000];
NN = [5000];
for N = NN
    random_angles = rand(N, 1) * 360;
    projections = radon(image, random_angles);
    for i = 2:N
        best_norm = 1000000;
        best_index = i;
        flipped = 0;
        for j = i:N
            if norm(projections(:, j) - projections(:, i-1)) < best_norm
                best_norm = norm(projections(:, j) - projections(:, i-1));
                best_index = j;
                flipped = 0;
            elseif norm(flip(projections(:, j)) - projections(:, i-1)) < best_norm
                best_norm = norm(flip(projections(:, j)) - projections(:, i-1));
                best_index = j;
                flipped = 1;
            end
        end
        if flipped == 1
            temp = projections(:, i);
            projections(:, i) = flip(projections(:, best_index));
            projections(:, best_index) = temp;
        else
            temp = projections(:, i);
            projections(:, i) = projections(:, best_index);
            projections(:, best_index) = temp;
        end
    end
    reconstructed_image = iradon(projections, 180/N , 'linear', 'Ram-Lak', 1, size(image, 1));
    RMSE = 1000000;
    rotation_angle = 0;
    flipped = 0;
    for angle = 0:1:359
        reconstructed_image_rotated = imrotate(reconstructed_image, angle, 'bilinear', 'crop');
        rmse = sqrt(mean((double(image(:)) - double(reconstructed_image_rotated(:))).^2));
        if(rmse < RMSE)
            RMSE = rmse;
            rotation_angle = angle;
            flipped = 0;
        end
        reconstructed_image_rotated = flip(reconstructed_image_rotated);
        rmse = sqrt(mean((double(image(:)) - double(reconstructed_image_rotated(:))).^2));
        if(rmse < RMSE)
            RMSE = rmse;
            rotation_angle = angle;
            flipped = 1;
        end
    end
    reconstructed_image = imrotate(reconstructed_image, rotation_angle, 'bilinear', 'crop');
    if flipped == 1
        reconstructed_image = flip(reconstructed_image);
    end
    RMSE = sqrt(mean((double(image(:)) - double(reconstructed_image(:))).^2));
    RMSE = RMSE / sqrt(mean(double(image(:)).^2));
    figure;
    imshow(reconstructed_image, []);
    title(['Reconstructed with N = ' num2str(N) ' and RMSE = ' num2str(RMSE)]);
    saveas(gcf, num2str(N), 'png');
end
