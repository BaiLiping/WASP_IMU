data = readtable("data/run2.txt");
t = data.Var1;
t = t - t(1);
t = t * 1e-3;
acc = [data.Var3, data.Var4];
plot(acc);
mean(acc)
%%

% Sample data: Create a noisy signal
Fs = 30; % Sampling frequency
t = 0:1/Fs:1; % Time vector
signal = acc(3091:3286,1);
signal = signal - mean(signal);

% Compute the FFT
Y = fft(signal);
L = length(signal); % Length of the signal

% Create a frequency vector
f = Fs*(0:(L/2))/L; % Frequency range

% Compute the two-sided spectrum and then the single-sided spectrum
P2 = abs(Y/L); % Two-sided spectrum
P1 = P2(1:L/2+1); % Single-sided spectrum
P1(2:end-1) = 2*P1(2:end-1); % Double the amplitude for non-DC components

% Plot the frequency spectrum
figure;
plot(f, P1);
title('Single-Sided Amplitude Spectrum of Signal');
xlabel('Frequency (f) [Hz]');
ylabel('|P1(f)|');
grid on;

%%

% Parameters
fc = 0.3;             % cutoff frequency (Hz)
fs = 30;           % sampling frequency (Hz)

% Example signal: sine at 1 Hz + sine at 20 Hz
x = acc(:,2);

% Compute filter coefficient (RC low-pass)
alpha = 2*pi*fc/fs;
alpha = alpha / (alpha + 1);

% Initialize output
y = zeros(size(x));
y(1) = x(1);

% Recursive filter
for n = 2:length(x)
    y(n) = y(n-1) + alpha*(x(n) - y(n-1));
end

% Plot
figure;
subplot(2,1,1); plot( x); title('Input signal');
subplot(2,1,2); plot( y); title('Filtered signal (low-pass)');

%%
t1 = t(1:1531);
seg1 = acc(1:1531,1);
t2 = t(1491:3067);
seg2 = acc(1491:3067,2);
t3 = t(3344:end);
seg3 = acc(3344:end, 1);


v1x = cumtrapz(t1,seg1 -0.8);
plot(t1, v1x);