
clear all,
clc;


img = imread('C:\Users\Administrator\Desktop\pic\O-HAZY\hazy\21_outdoor_hazy.jpg');  
H = rgb2gray(img);


subplot(1,2,2);  
imshow(img); title('original image');  

subplot(1,2,1);  

histogram(H);
title('histogram');  




