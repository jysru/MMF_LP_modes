%   modal computing from step index fiber   %
%   call the LPmodeStep.m function          %

clear all
close all

format long
warning off MATLAB:divideByZero   % il y a des divisions par zero dans le calcul de X (lorsque les fonctions
                                    %de Bessel du dénominateur ont des
                                    %zéros) mais le calcul continue ! :
                                    %suppression de l'avertissement

%
nCore=1.465 ;       % core index
nClad=1.445 ;       % clad index
Rcore=6.5e-6;        % core radius (m)
lambda=1.064e-6;    % wavelength

Nmaxmode=30;        % max mode number computed

Nsamp=128;           % number of spatial samples
SpLength=4*Rcore;   % numerical window dimension
step=SpLength/Nsamp;
x=-SpLength/2:step:SpLength/2-step;
y=x;

%______________ modes computed with the function LPmodeStep_______________%

[Nmode,Mode,coreBorder]=LPmodeStep(x,y,SpLength,nCore,nClad,Rcore,lambda,Nsamp,Nmaxmode);
%_________________________________________________________________________%


%%%%%%%% example of linear mode combining %%%%%%%%
SommeModes=zeros(Nsamp,Nsamp);
Et=zeros(Nsamp,Nsamp);
phi=2*pi*rand(Nmode,1);
amp=rand(Nmode,1);
alph=rand(Nmode,1);

Amp=NaN(max(cell2mat(Mode.M)+1),max(cell2mat(Mode.N)));
Phi=NaN(max(cell2mat(Mode.M)+1),max(cell2mat(Mode.N)));
for im=1:Nmode
    Et=sqrt(alph(im))*Mode.Ex{im}+sqrt(1-alph(im))*Mode.Ey{im};
    SommeModes=SommeModes+amp(im)*Et*exp(1i*phi(im));
    Amp(Mode.M{im}+1,Mode.N{im})=amp(im);
    Phi(Mode.M{im}+1,Mode.N{im})=phi(im);
end
figure()
%     imagesc(x*1e6,y*1e6,abs(SommeModes).^2))),                        % without core border on the image
    imagesc(x*1e6,y*1e6,abs(SommeModes).^2+0.2*coreBorder*max(max(abs(SommeModes).^2))), % with core border on the image
    title('Arbitrary linear mode combining'),xlabel('(µm)'), ylabel('(µm)'),axis 'equal',
figure()
    subplot(1,2,2),plot(x*1e6,abs(SommeModes(Nsamp/2+1,:).^2),'linewidth',2),
    xlabel('(µm)'),axis 'square', grid,
    title('cross section X')
    subplot(1,2,1),plot(y*1e6,abs(SommeModes(:,Nsamp/2+1).^2),'linewidth',2),
    xlabel('(µm)'),axis 'square', grid,
    title('cross section Y')
    
