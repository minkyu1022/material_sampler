% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A new style for the SPML Lab pre-prints
% Version: 0.1

% Modified based on the template from Matter Lab pre-prints
% Author: jiaru.bai@utoronto.ca
% Version: 0.1
% 
% Modified based on the template from FAIR Meta in their preprint: https://arxiv.org/abs/2503.03965
% A new style for FAIR pre-prints
% Author: dlp@meta.com
% Version: 1.1
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Class declarations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{spml_template}
\LoadClassWithOptions{article}

% Layout %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[top=2.25cm, bottom=2.5cm, left=2.5cm, right=2.5cm, columnsep=0.65cm]{geometry}
\RequirePackage{microtype}
\RequirePackage{placeins}
\RequirePackage{hyphenat}
\RequirePackage{setspace}
\RequirePackage{parskip}
\RequirePackage[latin, english]{babel}
\RequirePackage{lipsum}
\RequirePackage{etoolbox}

\RequirePackage{ifxetex}
\ifxetex\else
  \DisableLigatures[f]{family=sf*}
\fi

% Two-column text tuning: spaced paragraphs, comfortable line breaking.
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5\baselineskip}
\setlength{\emergencystretch}{1.5em}
\clubpenalty=10000
\widowpenalty=10000
\displaywidowpenalty=10000

% Graphics %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{graphicx}
\RequirePackage{subcaption}

% Tables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{booktabs}
\RequirePackage{nicematrix}
\RequirePackage{multirow}
\RequirePackage{bm}
\newcommand{\nm}[1]{#1}

% Colorful stuff %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[most]{tcolorbox}
\RequirePackage{xcolor}
% Mint palette:
% \definecolor{darkmint}{HTML}{0E1F1B}
% \definecolor{softmint}{HTML}{E8F5F2}
% \definecolor{highlightmint}{HTML}{5FC0AB}

% Blue palette:
% \definecolor{blue50}{HTML}{EAF0FD}
% \definecolor{blue100}{HTML}{C2D4F8}
% \definecolor{blue400}{HTML}{5384EC}
% \definecolor{blue900}{HTML}{142A5E}

% Vanilla palette
% \definecolor{vanillasoft}{HTML}{FDF8EE}
% \definecolor{vanillastrong}{HTML}{8C6D3F}
% \definecolor{vanillaink}{HTML}{2A1F0E}
% \AtBeginDocument{\color{vanillaink}}

% Purple palette
\definecolor{purplesoft}{HTML}{F6F2FA}
\definecolor{purplestrong}{HTML}{5B3A8C}
\definecolor{purplelink}{HTML}{7A53EC}
\definecolor{purpleink}{HTML}{231533}
\AtBeginDocument{\color{purpleink}}

% Rose palette
% \definecolor{rosesoft}{HTML}{FBF1F3}
% \definecolor{rosestrong}{HTML}{A33B52}
% \definecolor{roseink}{HTML}{2E1218}
% \AtBeginDocument{\color{roseink}}


% References %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{titletoc}
% \RequirePackage{minitoc}
\RequirePackage{hyperref}
\RequirePackage[noabbrev,nameinlink,nosort]{cleveref}
\hypersetup{
  colorlinks,
  linkcolor=purplelink,
  citecolor=purplelink,
  urlcolor=purpleink
}

% Default to authoryear
\DeclareOption{numbers}{\def\citestyleoption{numbers,square,sort&compress}}
\DeclareOption*{\PassOptionsToPackage{\CurrentOption}{natbib}}
\ProcessOptions\relax

% Load natbib with default option (authoryear) if nothing else is specified
\PassOptionsToPackage{round,authoryear}{natbib}
\RequirePackage{natbib}

% Apply user-specified citation style if provided
\ifdefined\citestyleoption
  \setcitestyle{\citestyleoption}
\fi

% font size small
\def\bibfont{\small}

% Section and caption format %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{titlesec}
%\titleformat*{\paragraph}{\itshape}
\titleformat{\section}
  {\Large\sffamily\bfseries\raggedright}
  {\thesection}{0.45em}{}
\titleformat{name=\section,numberless}
  {\Large\sffamily\bfseries\raggedright}
  {}{0pt}{}
\titleformat{\subsection}
  {\large\sffamily\bfseries\raggedright}
  {\thesubsection}{0.4em}{}
\titleformat{name=\subsection,numberless}
  {\large\sffamily\bfseries\raggedright}
  {}{0pt}{}
\titleformat{\subsubsection}
  {\normalsize\sffamily\bfseries\raggedright}
  {\thesubsubsection}{0.35em}{}
\titleformat*{\paragraph}{\normalfont\normalsize\bfseries}
\titlespacing*{\section}{0pt}{1.0em plus 0.25em minus 0.15em}{0.45em plus 0.1em}
\titlespacing*{\subsection}{0pt}{0.8em plus 0.2em minus 0.1em}{0.3em plus 0.08em}
\titlespacing*{\subsubsection}{0pt}{0.65em plus 0.15em minus 0.1em}{0.25em plus 0.05em}
\titlespacing*{\paragraph}{0pt}{0.5em plus 0.1em minus 0.05em}{0.55em}

\DeclareRobustCommand{\oldtextbf}[1]{{\bfseries #1}}
\DeclareTextFontCommand{\textbf}{\bfseries\sffamily}

\RequirePackage{caption}
\DeclareCaptionLabelSeparator{custom}{}
\DeclareCaptionFormat{custom}{{\sffamily\textbf{#1 #2}} #3}
\captionsetup{singlelinecheck=false,format=custom,labelsep=custom,font=small}
\captionsetup[sub]{singlelinecheck=true,format=custom,labelsep=custom,font=small}

% Fonts %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[T1]{fontenc}
\RequirePackage[semibold]{sourcesanspro}
\renewcommand*\ttdefault{cmvtt}

% Authors, affiliations, infos %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\renewcommand\addtolist[5][]{
  \begingroup
    \if\relax#3\relax\def\sep{}\else\def\sep{#5}\fi
    \let\protect\@unexpandable@protect 
    \xdef#3{\expandafter{#3}\sep #4[#1]{#2}}%
  \endgroup
}

\newcommand\authorlist{}
\newcommand\authorformat[2][]{{\sffamily \bfseries #2$^{#1}$}}
\renewcommand\author[2][]{\addtolist[#1]{#2}{\authorlist}{\authorformat}{, }}

\newcommand\affiliationlist{}
\newcommand\affiliationformat[2][]{{\normalsize $^{#1}$#2}}
\newcommand\affiliation[2][]{\addtolist[#1]{#2}{\affiliationlist}{\affiliationformat}{\\}}

\newcommand\contributionlist{}
\newcommand\contributionformat[2][]{{\small $^{#1}$#2}}
\newcommand\contribution[2][]{\addtolist[#1]{#2}{\contributionlist}{\contributionformat}{, }}

\newcommand\metadatalist{}
\newcommand\metadataformat[2][]{{\small {\sffamily \bfseries #1:} #2}}
\newcommand\metadata[2][]{\addtolist[#1]{#2}{\metadatalist}{\metadataformat}{\par}}

\renewcommand{\abstract}[1]{\def\abstractlist{{\color{purpleink} #1}}}
\newcommand{\email}[1]{\href{mailto:#1}{\texttt{#1}}}

\renewcommand\date[1]{\metadata[Date]{#1}}
\newcommand\correspondence[1]{\metadata[Correspondence]{#1}}

\renewcommand{\title}[1]{\newcommand{\titlelist}{{\huge\sffamily\bfseries #1}}}

\newcommand{\beginappendix}{\appendix{\huge\sffamily Appendix\par}}


% Titlebox %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


\newcommand{\mymaketitle}{%
  \tcbset{enhanced,frame hidden}
  \tcbset{left=0.5cm}
  \tcbset{right=0.5cm}
  \tcbset{top=0.5cm}
  \tcbset{bottom=0.5cm}
  \tcbset{arc=10pt}
  \tcbset{colback=purplesoft}
  \tcbset{before skip=0pt}
  \tcbset{grow to left by=1.5pt}
  \tcbset{grow to right by=1.5pt}
  \IfFileExists{spml_logo.png}{%
    \tcbset{overlay={\node[anchor=south east,inner sep=0pt] at ([xshift=-0.5cm,yshift=0.5cm]frame.south east){\includegraphics[width=1.15cm]{spml_logo.png}};}}%
  }{}
  \begin{tcolorbox}
    \setlength{\parindent}{0cm}
    \setlength{\parskip}{0.2cm}
    {
      \setlength{\parskip}{0cm}
      \raggedright
      \nohyphens
      %
      {
        \setstretch{1.618}
        \titlelist\par
      }
      \vskip 0.2cm
      %
      \authorlist\par
      \vskip 0.2cm
      %
      \affiliationlist\par
      %
      \contributionlist\par
    }
    %
    \abstractlist\par
    %
    \vskip 0.2cm
    %
    {
      \setlength{\parskip}{0cm}
      \ifdefempty{\metadatalist}{}{\metadatalist\par}
      % \ifdefempty{\metadatalist}{\vspace*{0.65cm}}{\metadatalist\par}
    }
  \end{tcolorbox}
  \tcbset{reset}
  \FloatBarrier
}

\if@twocolumn%
\renewcommand{\maketitle}{
\twocolumn[%
  \mymaketitle
  \vskip 0.38cm
]
}%
\else%
\renewcommand{\maketitle}{
  \mymaketitle
}%
\fi
