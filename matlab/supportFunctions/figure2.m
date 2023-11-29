function varargout = figure2(h, varargin)
  % FIGURE2(h, figVisible) More flexible figure command. Creates figure
  % window without the focus-theft if desired. Usage is identical to figure.
  %
  %
  % This script and its functions follow the coding style that can be
  % sumarized in:
  % * Variables have lower camel case
  % * Functions upper camel case
  % * Constants all upper case
  % * Spaces around operators
  %
  % Authors:  Néstor Uribe-Patarroyo
  %
  % NUP: 
  % 1. Wellman Center for Photomedicine, Harvard Medical School, Massachusetts
  % General Hospital, 40 Blossom Street, Boston, MA, USA;
  % <uribepatarroyo.nestor@mgh.harvard.edu>

  % MGH Flow Measurement project (v1.0)
  %
  % Changelog:
  %
  % V1.0 (2014-07-03): Initial version released
  %
  % Copyright Néstor Uribe-Patarroyo (2014)

if nargin == 2
  figVisible = varargin{1};
else
  figVisible = true;
end

if nargin >= 1 % We are being requested a particular figure handle
	if ishandle(h) % If it exist, just make it the current active figure
		set(0, 'CurrentFigure', h);
    if figVisible % We want it visible
      % Check if it's already visible, if not make it visible
      if ~get(h, 'visible')
        % Make it visible
        set(h, 'visible', 'on');
      end
    else % We want it invisible
      % Check if it's already invisible, if not make it invisible
      if get(h, 'visible')
        % Make it visible
        set(h, 'visible', 'off');
      end
    end
  else % We need to create new figure window
    % First make figures invisible by default, first ask current value
    defaultVisibility = get(0, 'defaultfigurevisible');
    set(0, 'defaultfigurevisible', 'off');
    % Now create the desired figure
    h = figure(h);
    % Now set properties I want
    %set(h, 'Renderer', 'OpenGL', 'RendererMode', 'manual');
    if figVisible % We want it invisible
      % Now make it visible
      set(h, 'visible', 'on');
    end
    % Go back to default 
    set(0, 'defaultfigurevisible', defaultVisibility);
    % Make the created window active
		set(0, 'CurrentFigure', h);
    if nargout == 1
      varargout{1} = h;
    end
	end
else % No handle specified, create next available
  % First make figures invisible by default, first ask current value
  defaultVisibility = get(0, 'defaultfigurevisible');
  set(0, 'defaultfigurevisible', 'off');
  % Now create the desired figure
  h = figure;
  % Now set properties I want
  %set(h, 'Renderer', 'OpenGL', 'RendererMode', 'manual');
  if figVisible % We want it invisible
    % Now make it visible
    set(h, 'visible', 'on');
  end
  % Go back to default
  set(0, 'defaultfigurevisible', defaultVisibility);
  % Make the created window active
  set(0, 'CurrentFigure', h);
  if nargout == 1
    varargout{1} = h;
  end
end