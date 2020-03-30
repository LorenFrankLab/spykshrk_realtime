% PROGRAM NAME: 	V8pre_goal_nowait
% AUTHOR: AKG
% DESCRIPTION: pretraining regime 	

% constants
int deliverPeriodBox= 150   	% how long to deliver the reward at home/center
int deliverPeriodOuter= 450   	% how long to deliver the reward at outer wells
int lockoutPeriod= 15000 	% length of lockout
int proxTime = 200 		% amount of time allowed to be away from nose poke
% variables
int rewardWell = 0
int currWell = 0
int lastWell = 0
int dio = 0
int homeCount = 0		% number of times rewarded at home
int centerCount = 0		% number of times rewarded at wait well
int locktype1 = 0 		% number of times lockout out by making other order error
int locktype2 = 0 		% number of times lockout from not holding in center
int trialtype = 0
int goalTotal = 0 % cumulative num outer visits
int otherCount = 0
int proximity = 0
int waslock = 0
int ripwait = 0			% state of waiting for rip
int riptime = 0			% track lenth of time intil rip 
int ripstart = 0		% starttime of ripwait
int replay_arm = 0
int taskState = 1 % 1 is cued trials 2 is content based trials

% initialize lights at home
portout[1] = 1
portout[2] = 0

portout[10] = 0
portout[11] = 0
portout[12] = 0
portout[13] = 0

;

% function to deliver reward to box wells
function 1
	portout[rewardWell]=1 % reward
	do in deliverPeriodBox 
		portout[rewardWell]=0 % reset reward
	end	
end;

% function to deliver reward to outer wells
function 2
	disp('outer reward')	
	portout[rewardWell]=1 % reward
	do in deliverPeriodOuter 
		portout[rewardWell]=0 % reset reward
	end	
end;

% Function to turn on output
function 3
	portout[dio]=1
end;

% function to turn off output
function 4	
	portout[dio]=0	
end;

%display status to scatesscript terminal and saved in sc log
function 5
	disp(homeCount)
	disp(centerCount)
	disp(locktype1)
	disp(locktype2)
	disp(goalTotal)
	disp(otherCount)
end;

function 6 % end lockout and reactivate home
	disp('WHITENOISE')	
	do in lockoutPeriod
		disp('LOCKEND')
	end
end;

% function send beep when large ripple is detected
function 7
	disp('large ripple trigger from spykshrk')
	if (ripwait == 1 && taskState==1) do
		disp('BEEP 1')
		ripwait = 0
		trialtype = 2	
		proximity = 0
		% this is format at add 10 msec delay
		do in 10
			disp('BEEP 2')
		end 	
	end
	portout[6]=1 % reward
	do in 30 
		portout[6]=0 % reset reward
	end		
end;

% function send beep when content is detected, replay_arm variable is updated at the same time
% we should be able to use the current value for replay_arm after disp('BEEP 1')
function 15
	disp('content trigger from spykshrk')
	if (ripwait == 1 && taskState==2) do
		disp(replay_arm)
		disp('BEEP 1')
		ripwait = 0
		trialtype = 2	
		proximity = 0
		% this is format at add 10 msec delay
		do in 10
			disp('BEEP 2')
		end 	
	end	
end;

% function to flip light in port 6
function 16
	disp('large ripple from spykshrk')
	portout[6]=1 % reward
	do in 300 
		portout[6]=0 % reset reward
	end	
end;



% CALLBACKS -- EVENT-DRIVEN TRIGGERS
callback portin[1] up % home well
	if trialtype != 3 do 
		currWell = 1
		disp('UP 1')
		waslock = 0
		disp(waslock)
	end
end;

callback portin[1] down
	if trialtype != 3 do
		lastWell = 1
		disp('DOWN 1')
	end
end;


callback portin[2] up % center well
	currWell = 2 % well currently active
	disp('UP 2')

	if (trialtype == 1 && lastWell==1) do
		proximity = 1
		disp('start rip wait')
		ripstart = clock()
		ripwait = 1
	end 
	if (trialtype==1 && lastWell==2 && proximity >0) do
		proximity=proximity+1
	end			
	if (trialtype != 3 && trialtype != 1 && currWell != lastWell && waslock != 1) do 
		disp('LOCKOUT 1')
	end

end;

callback portin[2] down
	lastWell=2 % well left, now last well
	disp('DOWN 2')
	% creates lockout 2 for not waiting
	if proximity>0 do
		do in proxTime	
			proximity=proximity-1	
			if (proximity <1 && trialtype <2) do
				disp('LOCKOUT 2')
				ripwait = 0
			end	
		end
	end
end;

% outer arm CALLBACKS


callback portin[10] up
	currWell = 10
	if currWell != lastWell do
		disp('UP 10')
	end
end;

callback portin[10] down
	lastWell = 10
	disp('DOWN 10')
end;

callback portin[11] up
	currWell = 11
	if currWell != lastWell do
		disp('UP 11')
	end
end;

callback portin[11] down
	lastWell = 11

	disp('DOWN 11')
end;

callback portin[12] up
	currWell = 12
	if currWell != lastWell do
		disp('UP 12')
	end
end;

callback portin[12] down
	lastWell = 12
	disp('DOWN 12')
end;

callback portin[13] up
	currWell = 13
	if currWell != lastWell do
		disp('UP 13')
	end
end;

callback portin[13] down
	lastWell = 13
	disp('DOWN 13')
end;

