

% 
[x,Fs] = audioread('C:\Users\tibor\OneDrive\Desktop\ISS\sentences\sx84.wav');

x = x - mean(x);

N = 512;

% rozdelenie segmentov o dlzke 25ms
wlen = 25e-3 * Fs;

% posunitie medzi ramcami
wshift = 10e-3*Fs; 

woverlap = wlen - wshift;
win = hamming(wlen); 

f = (0:(N/2)) / N * Fs;
t = (0:(1 + floor((length(x) - wlen) / wshift)))* wshift / Fs; % minus one as of Matlab ...
X = specgram (x, N, Fs, win, woverlap);

% vykreslenie
imagesc(t,f,10*log(abs(X).^2));
set (gca (), "ydir", "normal"); 
title("sx354.wav");
xlabel ("Time [s]"); 
ylabel ("Frequency [Hz]");

colormap(jet);




