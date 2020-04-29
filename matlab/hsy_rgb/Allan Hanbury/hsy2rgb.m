function [rout,g,b] = hsy2rgb(h,s,y)
%HSY2RGB Convert hue-saturation-luminance colors to red-green-blue.
%   M = HSY2RGB(H) converts an HSY color map to an RGB color map.
%   Each map is a matrix with any number of rows, exactly three columns,
%   and elements in the interval 0 to 1.  The columns of the input matrix,
%   H, represent hue, saturation and value, respectively.  The columns of
%   the resulting output matrix, M, represent intensity of red, blue and
%   green, respectively.
%
%   RGB = HSY2RGB(HSV) converts the HSY image HSY (3-D array) to the
%   equivalent RGB image RGB (3-D array).
%
%   As the hue varies from 0 to 1, the resulting color varies from
%   red, through yellow, green, cyan, blue and magenta, back to red.
%   When the saturation is 0, the colors are unsaturated; they are
%   simply shades of gray.  When the saturation is 1, the colors are
%   fully saturated; they contain no white component.  As the luminance
%   varies from 0 to 1, the brightness increases.
%
%   The colormap HSY is hsy2rgb([h s y]) where h is a linear ramp
%   from 0 to 1 and both s and v are all 1's.
%
%   See also RGB2HSV, COLORMAP, RGBPLOT.

%   Undocumented syntaxes:
%   [R,G,B] = HSY2RGB(H,S,Y) converts the HSY image H,S,Y to the
%   equivalent RGB image R,G,B.
%
%   RGB = HSY2RGB(H,S,Y) converts the HSY image H,S,Y to the 
%   equivalent RGB image stored in the 3-D array (RGB).
%
%   [R,G,B] = HSY2RGB(HSV) converts the HSY image HSY (3-D array) to
%   the equivalent RGB image R,G,B.
%

%   See Allan Hanbury's PhD thesis (CMM, Ecole des Mines de Paris, 2002).
%   Also, "The Taming of the Hue, Saturation and Brightness Colour Space",
%        Proceedings of the CVWW'02, Bad Aussee, Austria, 2002
%   Both these are available on http://www.prip.tuwien.ac.at/~hanbury


%   See Alvy Ray Smith, Color Gamut Transform Pairs, SIGGRAPH '78.
%   C. B. Moler, 8-17-86, 5-10-91, 2-2-92.
%   Copyright 1984-2001 The MathWorks, Inc. 
%   $Revision: 5.11 $  $Date: 2001/04/15 12:01:46 $

%   modified by Allan Hanbury 020805

if ( (nargin ~= 1) & (nargin ~= 3) )
  error('Wrong number of input arguments.');      
end

threeD = (ndims(h)==3); % Determine if input includes a 3-D array

if threeD
  s = h(:,:,2); y = h(:,:,3); h = h(:,:,1);
  siz = size(h);
  s = s(:); y = y(:); h = h(:);
elseif nargin==1 % HSV colormap
  s = h(:,2); y = h(:,3); h = h(:,1);
  siz = size(h);
else
  if ~isequal(size(h),size(s),size(y))
    error('H,S,V must all be the same size.');
  end
  siz = size(h);
  h = h(:); s = s(:); y = y(:);
end

%Calculate the inverse conversion matrix
A=[0.2125 0.7154 0.0721; 1 -0.5 -0.5; 0 -sqrt(3)/2 sqrt(3)/2];
Ai=inv(A);

%convert h to radians
h=h./180.0;
h=h.*pi;


%chroma C
k=h./(pi/3.0);
k=floor(k);
hstar=h - k.*(pi/3.0);
Ct=(sqrt(3.0).*s);
Cb=(2.0.*sin(-hstar + (2.0*pi/3.0)));
C=Ct./Cb;

%C1 and C2
C1=C.*cos(h);
C2=-C.*sin(h);

%R,G,B
r=Ai(1,1).*y + Ai(1,2).*C1 + Ai(1,3).*C2;
g=Ai(2,1).*y + Ai(2,2).*C1 + Ai(2,3).*C2;
b=Ai(3,1).*y + Ai(3,2).*C1 + Ai(3,3).*C2;

%sort out some possible small rounding errors
indic=find(r>1);
r(indic)=1;
indic=find(g>1);
g(indic)=1;
indic=find(b>1);
b(indic)=1;
indic=find(r<0);
r(indic)=0;
indic=find(g<0);
g(indic)=0;
indic=find(b<0);
b(indic)=0;

% h = 6*h(:);
% k = fix(h-6*eps);
% f = h-k;
% t = 1-s;
% n = 1-s.*f;
% p = 1-(s.*(1-f));
% e = ones(size(h));
% r = (k==0).*e + (k==1).*n + (k==2).*t + (k==3).*t + (k==4).*p + (k==5).*e;
% g = (k==0).*p + (k==1).*e + (k==2).*e + (k==3).*n + (k==4).*t + (k==5).*t;
% b = (k==0).*t + (k==1).*t + (k==2).*p + (k==3).*1 + (k==4).*1 + (k==5).*n;
% f = y./max([r(:);g(:);b(:)]);

if nargout<=1
  if (threeD || nargin==3)
    rout = zeros([siz,3]);
    rout(:,:,1) = reshape(r,siz);
    rout(:,:,2) = reshape(g,siz);
    rout(:,:,3) = reshape(b,siz);
  else
    rout = [r g b];
  end
else
  rout = reshape(r,siz);
  g = reshape(g,siz);
  b = reshape(b,siz);
end
