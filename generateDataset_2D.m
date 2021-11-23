clear
clc

shape_size = 80; % # of pixels for a shape, each shape is an image
nSample = 10000;    % # of samples



%%%%%%%%%%%%%%% simulation parameters %%%%%%%%%%%%%%%%
K_gauss = shape_size/15;
p = 0.5;
margin = ceil(K_gauss);
K_fuse = shape_size/5;
aniso_shrink_rate = 0.1;
heat_level = 0.5;

%%%%%%%%%%%%%%% make folder %%%%%%%%%%%%%%%%
mkdir('dataset')
mkdir('dataset/distorted')
mkdir('dataset/original')


for ii = 1:nSample
    %%%%%%%%%%%%%%% original %%%%%%%%%%%%%%%%
    firstLayer=randi(2,shape_size,shape_size)-1;
    cc=zeros(shape_size+margin*2,shape_size+margin*2);
    cc((margin+1):(margin+shape_size),(margin+1):(margin+shape_size))=firstLayer;
    cc=imgaussfilt(cc,K_gauss);
    firstLayer=cc((margin+1):(margin+shape_size),(margin+1):(margin+shape_size));
    shape=firstLayer>p;

    CC = bwconncomp(shape,8);
    numPixels = cellfun(@numel,CC.PixelIdxList);
    [biggest,idx] = max(numPixels);
    shape_new = zeros(size(shape));
    shape_new(CC.PixelIdxList{idx}) = shape(CC.PixelIdxList{idx});

    shape_new = (imgaussfilt(shape_new,5)>0.5)*1.0;

    %%%%%%%%%%%%%%% distortion %%%%%%%%%%%%%%%%
    HeatDefused = imgaussfilt(shape_new,K_fuse);
    sdf = bwdist(shape_new==0,'euclidean'); % signed distance function

    out = sdf>(HeatDefused.^2*shape_size*aniso_shrink_rate);

    col = zeros(shape_size,shape_size,3);
    col(:,:,1) = shape_new;
    col(:,:,2) = out;
    
%     imshow(col)

    % use this to format the file name to include leading zeros. This 
    % ensures that the order of the files does not get mixed up. It will 
    % work for up to 99,999 samples. Increase the constant '5' below for 
    % larger sample sizes.
    strPadded = sprintf('%05d', ii);

    %%%%%%%%%%%%%%% write image %%%%%%%%%%%%%%%%
    imwrite(shape_new,['dataset/original/' strPadded '.png']);
    imwrite(out,['dataset/distorted/' strPadded '.png']);
end
