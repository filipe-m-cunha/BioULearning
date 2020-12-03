using PyPlot

# 13/3/2017                                          DANNY NISSANI [NISSENSOHN]

#               HUL_CLASSIFIER_DYNAMICS_PLOT_SCRIPT_130317
#               -------------------------------------------

# PLOT MODEL DYNAMICAL VARIABLES
#-------------------------------
if Display_Dynamics .== 1

    if ss % Dynamics_Plot_Period == 0 || ss == 1 || (ss .== R_start + 300)
        # PLOTTED NEURON SELECTION OUT OF Any_Separate_List
        if ss .== 1

            th_pr = theta_N

            w_pr = w_N

        end

        if ss .== R_start + 300 # IDENTIFY MAX CHANGE theta, w [ONE TIME ONLY]

            [d_th_mx, th_indx] = max(abs(theta_N - th_pr)); # INDEX OF NEURON 
            # WITH MAX theta CHANGE
            [d_w_mx, w_indx] = max(sum(abs(w_N - w_pr))); # INDEX OF NEURON 
            # WITH MAX w CHANGE 
            if sum(sum(abs(w_N - w_pr))) .== 0 # FOR EXPLORE
                
                keyboard()
                
            end
             
            w_pr = nothing

        end

        if ss >= R_start + 300 #PLOT MAX CHANGE IDENTIFIED theta; w VARS
            
            figure(1); 
            subplot(2, 1, 1);
            plot(ss, theta_N[th_indx], "ob"); 
            legend("SELECTED NEURON theta DYNAMICS");            

            w_N_m = sum(abs(w_N[:, w_indx])) / d; # A METRIC TO TRACK CHANGE OF SELECTED
            # NEURON WEIGHT VECTOR
            subplot(2, 1, 2);
            plot(ss, w_N_m, "+k"); 
            legend("SELECTED NEURON norm-1[w] DYNAMICS")

        end

    end

end


