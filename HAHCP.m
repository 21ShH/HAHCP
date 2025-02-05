
clc;
clear;


img = imread('C:\Users\Administrator\Desktop\pic\O-HAZY\hazy\21_outdoor_hazy.jpg');       


I=double(img)/255; 
[m,n]=size(I,1,2);
subplot(2,3,1);
imshow(I,[]);
title('original image');

a = 0.82;
b = 1 - a;

gray = rgb2gray(I);


%% dark channel img
id=zeros(m,n);
for i=1:m
    for j=1:n
        id(i,j)=min(I(i,j,:));
    end
end
dark_img = ordfilt2(id,1,ones(5,5),'symmetric');           
subplot(2,3,2);
imshow(dark_img);
title('dark image');

%% bright channel img
ib=zeros(m,n);
for i=1:m
    for j=1:n
        ib(i,j)=max(I(i,j,:));
    end
end
bright_img = ordfilt2(ib,25,ones(5,5),'symmetric');  
subplot(2,3,3);
imshow(bright_img);
title('bright image');

%% local A
s = strel('disk',12);
A_local = imclose(bright_img,s);

%% global  A
dark_channel = dark_img;
bright_channel = bright_img;

%A_dark = max(max(dark_channel))*0.999;
A_dark = Adark_estimate(dark_channel,img);
%A_bright= min(min(bright_channel))*0.999;
A_bright = Abright_estimate(bright_channel,img);

A_global = a*A_dark + b*A_bright;

%% A

A = 0.5*A_global + 0.5*A_local;


%% t
w0=0.95;
t_dark =  1 - w0 * dark_img/ A_dark;
t1 = max(0.4,t_dark);

t_bright = (w0*bright_img - A_bright)/(1-A_bright);
t2 = max(0,t_bright);

t_temp = a* t1 + b* t2;
              
t = guidedfilter(gray,t_temp,3,0.01);
subplot(2,3,4);
imshow(t);
title('transmission');

%% dehaze
I_out=zeros(m,n,3);
for k=1:3
    for i=1:m
        for j=1:n
            I_out(i,j,k)=(I(i,j,k)-A(i,j))/t(i,j)+A(i,j);
        end
    end
end

subplot(2,3,5);
imshow(I_out,[]);
title('dehaze image');





r = double(I_out(:,:,1));
g = double(I_out(:,:,2));
b = double(I_out(:,:,3));

[m,n] = size(r);

r_log = log(r+1);
g_log = log(g+1);
b_log = log(b+1);

Rfft = fft2(r);
Gfft = fft2(g);
Bfft = fft2(b);

sigma1 = 100;
sigma2 = 225;
sigma3 = 350;

f1 = fspecial('gaussian', [m, n], sigma1);
f2 = fspecial('gaussian', [m, n], sigma2);
f3 = fspecial('gaussian', [m, n], sigma3);

efft1 = fft2(double(f1));
efft2 = fft2(double(f2));
efft3 = fft2(double(f3));

r1 = ifft2(Rfft.*efft1);
g1 = ifft2(Gfft.*efft1);
b1 = ifft2(Bfft.*efft1);

r1_log = log(r1 + 1);
g1_log = log(g1 + 1);
b1_log = log(b1 + 1);

R1 = r_log - r1_log;
G1 = g_log - g1_log;
B1 = b_log - b1_log;

r2 = ifft2(Rfft.*efft2);
g2 = ifft2(Gfft.*efft2);
b2 = ifft2(Bfft.*efft2);

r2_log = log(r2 + 1);
g2_log = log(g2 + 1);
b2_log = log(b2 + 1);

R2 = r_log - r2_log;
G2 = g_log - g2_log;
B2 = b_log - b2_log;

r3 = ifft2(Rfft.*efft3);
g3 = ifft2(Gfft.*efft3);
b3 = ifft2(Bfft.*efft3);

r3_log = log(r3 + 1);
g3_log = log(g3 + 1);
b3_log = log(b3 + 1);

R3 = r_log - r3_log;
G3 = g_log - g3_log;
B3 = b_log - b3_log;

R = R1/3 + R2/3 + R3/3;
G = G1/3 + G2/3 + G3/3;
B = B1/3 + B2/3 + B3/3;
% R = 0.1*R1 + 0.3*R2 +0.6* R3;
% G = 0.1*G1 + 0.3*G2 +0.6* G3;
% B = 0.1*B1 + 0.3*B2 +0.6* B3;


R = exp(R);
R_min = min(min(R));
R_max = max(max(R));

G = exp(G);
G_min = min(min(G));
G_max = max(max(G));

B = exp(B);
B_min = min(min(B));
B_max = max(max(B));



R = (R - R_min)/(R_max - R_min);
G = (G - G_min)/(G_max - G_min);
B = (B - B_min)/(B_max - B_min);


IO = cat(3, R, G, B);

 subplot(2,3,6); 
 imshow(IO);
 title('HAHCP_img');
 imwrite(IO,'aresult.jpg');



function  [Ac]=Adark_estimate(dark,img)
    img=im2double(img);  
    R1=img(:,:,1);
    G1=img(:,:,2);
    B1=img(:,:,3);
    [a,b]=size(dark);   
    c=ceil(a*b/1000);       
    r1=zeros(c,1);     
    g1=zeros(c,1);
    b1=zeros(c,1);
    m=0.9;
    x=1;
    d(1,1)=0;
    q(1,1)=0;
while size(d,1)<=c
    for i=1:a
    for j=1:b
        if dark(i,j)>m && size(d,1)<=c
            d(x,1)=i;
            q(x,1)=j;
            x=x+1;
        end
        if size(d,1)>c
            break 
        end
    end
    end
        if size(d,1)<=c
            m=m-0.1;
        end
end
    for p=1:c
        r1(p,1)=R1(d(p,1),q(p,1));
        g1(p,1)=G1(d(p,1),q(p,1));
        b1(p,1)=B1(d(p,1),q(p,1));
    end
        Ar=max(max(r1));
        Ag=max(max(g1));
        Ab=max(max(b1));
        
        Ac=(Ar+Ag+Ab)/3;
end

function  [Ac]=Abright_estimate(bright,img)
    img=im2double(img);  
    R1=img(:,:,1);
    G1=img(:,:,2);
    B1=img(:,:,3);
    [a,b]=size(bright);   
    c=ceil(a*b/1000);       
    r1=zeros(c,1);     
    g1=zeros(c,1);
    b1=zeros(c,1);
    m=0.9;
    x=1;
    d(1,1)=0;
    q(1,1)=0;
while size(d,1)<=c
    for i=1:a
    for j=1:b
        if bright(i,j)>m && size(d,1)<=c
            d(x,1)=i;
            q(x,1)=j;
            x=x+1;
        end
        if size(d,1)>c
            break 
        end
    end
    end
        if size(d,1)<=c
            m=m-0.1;
        end
end
    for p=1:c
        r1(p,1)=R1(d(p,1),q(p,1));
        g1(p,1)=G1(d(p,1),q(p,1));
        b1(p,1)=B1(d(p,1),q(p,1));
    end
        Ar=min(min(r1));
        Ag=min(min(g1));
        Ab=min(min(b1));
        
        Ac=(Ar+Ag+Ab)/3;
end

function q = guidedfilter(I, p, r, eps)

    [h, w] = size(I);
    N = boxfilter(ones(h, w), r); 
    mean_I = boxfilter(I, r) ./ N;
    mean_p = boxfilter(p, r) ./ N;
    mean_Ip = boxfilter(I.*p, r) ./ N;
    cov_Ip = mean_Ip - mean_I .* mean_p; 

    mean_II = boxfilter(I.*I, r) ./ N;
    var_I = mean_II - mean_I .* mean_I;

    a = cov_Ip ./ (var_I + eps); 
    b = mean_p - a .* mean_I; 

    mean_a = boxfilter(a, r) ./ N;
    mean_b = boxfilter(b, r) ./ N;

    q = mean_a .* I + mean_b; 
end


function imDst = boxfilter(imSrc, r)

 
[hei, wid] = size(imSrc);
imDst = zeros(size(imSrc));
 
imCum = cumsum(imSrc, 1);
imDst(1:r+1, :) = imCum(1+r:2*r+1, :);
imDst(r+2:hei-r, :) = imCum(2*r+2:hei, :) - imCum(1:hei-2*r-1, :);
imDst(hei-r+1:hei, :) = repmat(imCum(hei, :), [r, 1]) - imCum(hei-2*r:hei-r-1, :);
 
%cumulative sum over X axis
imCum = cumsum(imDst, 2);
%difference over Y axis
imDst(:, 1:r+1) = imCum(:, 1+r:2*r+1);
imDst(:, r+2:wid-r) = imCum(:, 2*r+2:wid) - imCum(:, 1:wid-2*r-1);
imDst(:, wid-r+1:wid) = repmat(imCum(:, wid), [1, r]) - imCum(:, wid-2*r:wid-r-1);
end
function T = OTSU(hist)
   sum_num = sum(hist);
   pro = hist./sum_num;
   T = 1;
   delta_max = 0;
   temp = [];
   for i = 1:256
       w0 = 0;w1 = 0;u0_temp = 0;u1_temp = 0;u0 = 0;u1 = 0;
delta_temp = 0;
       for j=1:256
           if(j<=i)
               w0 =  pro(j) + w0;
               u0_temp = j*pro(j)+u0_temp;
           else
               w1 = w1+pro(j);
               u1_temp = u1_temp + j*pro(j);
           end
       end 
       if(w0>0)
         u0 = u0_temp / w0;
       end
       if(w1>0)
       u1 = u1_temp / w1;
       end
       delta_temp = w0*w1*(u0-u1)^2;
       temp = [temp;delta_temp];
       if(delta_temp > delta_max)
          delta_max = delta_temp;
            T = i;
       end
   end
   T = T-1;
end