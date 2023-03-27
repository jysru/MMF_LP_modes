%       *************************************************
%       *     CALCUL DE LA CONSTANTE DE PROPAGATION     *
%       *         DES MODES VRAIS (TE, TM, HE, EH)      *
%       *              DES FIBRES A SAUT                *
%       *   ET DE LA DISTRIBUTION DE CHAMP DE CES MODES *
%       *     (Dominique - NOVEMBRE 2007)               *
%       *************************************************

function [Nmode,Mode,coreBorder]=LPmodeStep(x,y,SpLength,nCore,nClad,Rcore,lambda,Nsamp,Nmaxmode)

NA=sqrt(nCore^2-nClad^2) ; %ouverture numérique

celer=3e8;
omeg=2*pi*celer/lambda;
mu0=4*pi*1e-7;
epsilon0=1e-9/(36*pi);
epsC=epsilon0*(nCore^2);
epsG=epsilon0*(nClad^2);
k0=2*pi/lambda;
imode=1;

V=2*pi*Rcore*NA/lambda;
% DEFINITION DE LA ZONE DE CALCUL AUTOUR DU CENTRE DE LA FIBRE 
% ET DE LA RESOLUTION pour le calcul des champs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[X,Y]= meshgrid(x,y);

epsi=0.04E-6;
indcoeur=find(and((X.^2+Y.^2)<=(Rcore+epsi)^2,(X.^2+Y.^2)>=(Rcore-epsi)^2));
coreBorder=zeros(Nsamp,Nsamp);
coreBorder(indcoeur)=1;

% CALCULS DES VARIABLES D'ESPACE R ET THETA
R=sqrt(X.^2+Y.^2);
% theta=atan2(Y,X);
theta=atan2(Y,X);
%
Nmax=[];
%   CHOIX DU MODE CONSIDERE
%    si on veut TE0n  : écrire : test_mode=1    M=0  et  N=n >1  -->(mode LP1n)
%    si on veut TM0n  : écrire : test_mode=2    M=0   et  N=n >1  -->(mode LP1n)
%    si on veut HEm,n  : écrire : test_mode=3   M=m >=1  et  N=2n-1 (donc impair)  -->(mode LPm-1 n)
%    si on veut EHm,n  : écrire test_mode=4   M=m >=1  et  N=2n (donc pair) -->(mode LPm+1 n)

for M=1:Nmaxmode   %premier indice des modes EH considérés, borne supérieure indicative (arret lorsqu'il n'y a plus de solution à l'équation de dispersion)
 Nn=1;  % N deuxième indice des modes TE TM EH ou HE considérés, ici sa valeur initiale

         
  U=linspace(0,V,1000) ;          % matrice des ctes de prop transverses dans le coeur, entre 0 et V
  W=sqrt(V^2-U.^2) ;             % matrice des ctes de prop transverses dans la gaine, entre 0 et V
  JPRIMU=0.5*(besselj(M-1,U)-besselj(M+1,U));        % dérivée de Jm(U)
  KPRIMW=-0.5*(besselk(M-1,W)+besselk(M+1,W));       % dérivée de Km(W)
  F1=JPRIMU./(U.*besselj(M,U))+KPRIMW./(W.*besselk(M,W));
  CO=nCore^2/nClad^2;
  F2=CO*JPRIMU./(U.*besselj(M,U))+KPRIMW./(W.*besselk(M,W));    % eq de dispersion complète: F1*F2=m^2*F3*F4
  F3=1./(U.^2)+1./(W.^2);
  F4=CO./(U.^2)+1./(W.^2);
  
   %  cas M différent de 0 (modes EH ou HE)
    % MODES EH ou HE
    Xx=(F1.*F2)-(M*M)*(F3.*F4);   % eq de dispersion dont on cherche les zeros pour les modes HE ou EH
                            % pour la nème sol : mode EHm, n/2 si n est pair
                            % et mode HEm, (n+1)/2 si n est impair
  DECALX=circshift(Xx,[0,-1]); % = matrice X dont les lignes sont décalées de 0 cases (mais y'a qu'une ligne, donc ça ne fait rien)
                               % et les colonnes sont décalées d'une case vers la gauche
  TEST =Xx.*DECALX   ;    % quand TEST<0, c'est qu'il y a changement de signe de la fonction X juste après ce X
         BOUT=size(U,2);  % BOUT est l'indice du dernier élément de la matrice U (=taille de U)
         TEST(BOUT)=1;    % on met arbitrairement le dernier elt de TEST à 1 (>0)car le signe<0 de TEST sur le dernier elt 
                           % n'est pas l'indication d'un passage par 0 de X
  INDI=find( TEST < 0 );  % donne les indices de la matrice X juste avant le changement de signe
                           % remarque :pour les modes EH et HE il n'y a pas de branche infinie de X et on garde donc
                           % tous les passages par 0
NBVAL=size(INDI,2);   % donne la taille de la matrice ligne "INDI", c'est à dire le nombre de passages par zéro trouvés pour X
                       % càd le nombre de modes EHm... et HEm... guidés pour ce V 
 Nmax=[Nmax (NBVAL+1)/2]; % stockage des valeurs de l'indice Nmax
     %
for Nn=1:(NBVAL+1)/2  % si condition pas vérifiée, le Nieme mode EH ou HE n'est pas guidé pour le V considéré
    N=2*Nn-1; % Modes LP (M-1),Nn
     U=linspace(0,V,1000) ; % matrice des ctes de prop transverses dans le coeur, entre 0 et V
    Uinf=U(INDI(N));        % Uinf est la valeur de U juste avant le Nième passage par zéro de X
    Usup=U(INDI(N)+1);      %  Usup est la valeur de U juste avant le Nième passage par zéro de X
     while abs((Uinf-Usup)/Usup) > 10e-10     % test sur la précision de détermination du U solution
         U2=linspace(Uinf,Usup,100);  % précision sur Usolution insuffisante : on re-échantillonne l'intervalle entre Usup et Uinf
         W2=sqrt(V^2-U2.^2);          % calcul de la matrice W2 associée à la nouvelle matrice U2
         JPRIMU2=0.5*(besselj(M-1,U2)-besselj(M+1,U2));
         KPRIMW2=-0.5*(besselk(M-1,W2)+besselk(M+1,W2));
         F12=JPRIMU2./(U2.*besselj(M,U2))+KPRIMW2./(W2.*besselk(M,W2));
         F22=CO*JPRIMU2./(U2.*besselj(M,U2))+KPRIMW2./(W2.*besselk(M,W2));
         F32=1./(U2.^2)+1./(W2.^2);
         F42=CO./(U2.^2)+1./(W2.^2);
         X2=F12.*F22-(M*M)*(F32.*F42); % eq de disp des modes EH ou HE dans l'intervalle [Uinf, Usup] où l'on sait qu'il y a le Nième passage par zéro que l'on cerne 
         DECALX2=circshift(X2,[0,-1]); 
         TEST2 =X2.*DECALX2  ;         % comme plus haut, TEST2<0 indique que la fonction X2 change de signe juste après      
         BOUT=size(U2,2);      % BOUT = nbre d'elts de la matrice U2 = indice du dernier élément de U2
         TEST2(BOUT)=1;     % comme plus haut, artifice pour ne pas prendre en compte le possible changement de signe signalé par le dernier elt de TEST2
         INDI2=find( TEST2 < 0 ); % INDI2 donne l'indice de l'elément de la matrice X2 juste avant le changement de signe (il n'y en a qu'un ici--> INDI2 est un entier unique):
         Uinf=U2(INDI2);          %Uinf et Usup sont les deux nouvelles valeurs de U cernant le zéro de X, et on re-test la précision avec le while
         Usup=U2(INDI2+1);
     end                    % fin du test de précision sur U (while)
%             if mod(N,2)==0;%test de parité sur N
%                 mm=M+1;
%                 nn=N/2;
%             else
%                 mm=M-1;
%                 nn=N/2+1;
%             end
         
     Bnorm=1-(Uinf/V)^2;       % Bnorm = constante de propag normalisée du mode EHm,n/2 ou HEm,(n+1)/2, associée à V
     Cprop(M,Nn)=k0*sqrt(nClad^2+NA^2*Bnorm);% constante de propagation
% end       % fin de la boucle sur N pour les modes HEm... ou EHm..., pour M diff de 0, et pour le V considéré
% end       %  fin de boucle sur M  

% CALCULS DES CHAMPS

% CALCUL DES CONSTANTES A B C ET D DES EXPRESSIONS DES CHAMPS 
% OBTENUES EN ECRIVANT LA CONTINUITE DES COMPOSANTES TANGENTIELLES
% Ez, Etheta, Hz et Htheta A L'INTERFACE COEUR GAINE (R=a)

U=Uinf;
W=sqrt(V^2-U^2);
beta=sqrt((k0*nCore)^2-(U/Rcore)^2);

JPRIMU=0.5*(besselj(M-1,U)-besselj(M+1,U));        % dérivée de Jm(U)
KPRIMW=-0.5*(besselk(M-1,W)+besselk(M+1,W));
JU=besselj(M,U);
KW=besselk(M,W);
USURA=U/Rcore;
WSURA=W/Rcore;
X1=JU;
X2=KW;
X3=(beta*M*JU)/(USURA^2*Rcore);
X4=omeg*mu0*JPRIMU/USURA;
X5=beta*M*KW/(WSURA^2*Rcore);
X6=omeg*mu0*KPRIMW/WSURA;
X7=omeg*epsC*JPRIMU/USURA;
X8=(beta*M*JU)/(USURA^2*Rcore);
X9=omeg*epsG*KPRIMW/WSURA;
X10=beta*M*KW/(WSURA^2*Rcore);

%  %mode EH ou HE
    A=1;
    C=A*JU/KW;
    Z1=X3+(X1/X2)*X5;
    Z2=X4+(X1/X2)*X6;
    Z3=X7+(X1/X2)*X9;
    Z4=X8+(X1/X2)*X10;
    B=A*(Z1/Z2);
    %  on peut aussi calculer B=A*(Z3/Z4) : meme resultat qu'au dessus
    D=A*(Z1/Z2)*(X1/X2);
    %  on peut aussi calculer D=A*(Z3/Z4)*(X1/X2)  : meme resultat qu'au dessus
%
% ----------------    CALCUL DES CHAMPS      ------------------                 
%
% EXPRESSIONS DES CHAMPS DANS LE COEUR
JM=besselj(M,U*R/Rcore);
JPRIMM=0.5*(besselj(M-1,U*R/Rcore)-besselj(M+1,U*R/Rcore));
  %mode EH ou HE
%     Ezc=A*JM.*sin(M*theta);
    Erc=(-A*beta*JPRIMM/USURA+B*omeg*mu0*(M./R).*JM./(USURA^2)).*sin(M*theta);
    Ethetac=(-A*beta*(M./R).*JM./(USURA^2)+B*omeg*mu0.*JPRIMM/USURA).*cos(M*theta);
%
% EXPRESSIONS DES CHAMPS DANS LA GAINE
%
KM=besselk(M,W*R/Rcore);
KPRIMM=-0.5*(besselk(M-1,W*R/Rcore)+besselk(M+1,W*R/Rcore));
 %mode EH ou HE
%     Ezg=C*KM.*sin(M*theta);
    Erg=(C*beta*KPRIMM/WSURA-D*omeg*mu0*(M./R).*KM/(WSURA^2)).*sin(M*theta);
    Ethetag=(C*beta*(M./R).*KM/(WSURA^2)-D*omeg*mu0*KPRIMM/WSURA).*cos(M*theta);
%
% MISE A ZEROS DES ELTS DES MATRICES 'CHAMPS DANS LE COEUR' ET 'CHAMPS DANS LA GAINE' 
% LA OU LES EXPRESSIONS QUI LES ONT CALCULES NE SONT PAS VALIDES
%
for II=1:Nsamp
   for JJ=1:Nsamp
      if R(II,JJ)>=Rcore
         Erc(II,JJ)=0;
         Ethetac(II,JJ)=0;
      else
         Erg(II,JJ)=0;
         Ethetag(II,JJ)=0;
      end
   end
end

Etheta=Ethetac+Ethetag;
Er=Erc+Erg;

  if M==1
        Ex=Er.*sin(theta)+Etheta.*cos(theta);   
  else
        Ex=Er.*cos(theta)-Etheta.*sin(theta);   
  end
  
Ey=Er.*sin(theta)+Etheta.*cos(theta);

% il manque la valeur du pixel central - c'est deux lignes pour y remédier
% de façon rustique
Ex(Nsamp/2+1,Nsamp/2+1)=(Ex(Nsamp/2,Nsamp/2+1)+Ex(Nsamp/2+2,Nsamp/2+1)+Ex(Nsamp/2+1,Nsamp/2)+Ex(Nsamp/2+1,Nsamp/2+2))/4;
Ey(Nsamp/2+1,Nsamp/2+1)=(Ey(Nsamp/2,Nsamp/2+1)+Ey(Nsamp/2+2,Nsamp/2+1)+Ey(Nsamp/2+1,Nsamp/2)+Ey(Nsamp/2+1,Nsamp/2+2))/4;

betaMode(imode)=k0*sqrt(nClad^2+NA^2*Bnorm);

mode.M{imode}=M-1;
mode.N{imode}=Nn;
mode.beta{imode}=betaMode(imode);
mode.Ex{imode}=Ex;
mode.Ey{imode}=Ey;

    figure()
    imagesc(x*1e6,y*1e6,abs(Ey).^2) % without core border on the image
    imagesc(x*1e6,y*1e6,abs(Ey).^2+0.2*coreBorder*max(max(abs(Ey).^2))) % with core border on the image
    xlabel('en (µm)'), ylabel('en µm'),axis 'equal',
    title(['LP ',num2str(M-1),num2str(Nn),' (Ey)']) 
    
    figure()
    imagesc(x*1e6,y*1e6,abs(Ex).^2) % without core border on the image
    imagesc(x*1e6,y*1e6,abs(Ex).^2+0.2*coreBorder*max(max(abs(Ex).^2))) % with core border on the image
    xlabel('en (µm)'), ylabel('en µm'),axis 'equal',
    title(['LP ',num2str(M-1),num2str(Nn),' (Ex)'])
    
    figure()
    plot(x*1e6,abs(Ex(:,Nsamp/2+1).^2),'linewidth',2),
    xlabel('(µm)'),axis 'square', grid,
    title(['Cross section Y - LP ',num2str(M-1),num2str(Nn),' (Ex)'])

imode=imode+1;
    end      % fin de la boucle sur N pour les modes HEm... ou EHm..., pour M diff de 0, et pour le V considéré
end %  fin de boucle sur M  

Nmode=imode-1;

[blah, order] = sort(betaMode,'descend'); % modes triés dans l'ordre décroissant des beta

Mode.M=mode.M(order);
Mode.N=mode.N(order);
Mode.beta=mode.beta(order);
Mode.Ex=mode.Ex(order);
Mode.Ey=mode.Ey(order);
save('Mo.mat','Mode');

fiber.lambda=lambda;
fiber.ncore=nCore;
fiber.nclad=nClad;
fiber.rcore=Rcore;
fiber.Nbpix=Nsamp;
fiber.Nmode=Nmode;
fiber.Windowlength=SpLength;
save('Fiber.mat','fiber');

end
