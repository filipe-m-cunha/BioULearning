# 6/4/2017                                          DANNY NISSANI [NISSENSOHN]

#         NC_IMAGENET_HUL_1vsALL_NC_Assgn_Top_n_Err_Est_SCRIPT_060417
#         -----------------------------------------------------------

# THIS SCRIPT MAY BE EXECUTED INDEPENDENTLY; OR INVOKED BY 
# NC_IMAGENET_HUL_Multi_Epoch_Classifier_230317; IF INDEPENDENTLY WE SHOULD
# FIRST LOAD THE DESIRED multi_epoch_classifier_results_filename FILE
# BASED UPON
# NC_IMGNT_RSNT152_HUL_1vsALL_NC_Assgn_Top_n_Err_Est_SCRIPT_270317:
# 1. ONLY POS VARIABLES CALCULATED; ALL NEG OMITTED
# 2. OPTIMAL Lambda FOR Perr AND P_err_Top_n INCLUDED
# BASED ALSO UPON: 
# NC_IMGNT_RSNT152_HUL_One_vs_All_NC_Assgn_Err_Est_SCRIPT_070317:
# a. SOFTMAX ACTIVATION FUNCTION [INSTEAD SEMI-LINEAR], AND 
# b. TOP-n Perr ESTIMATE

# THE SCRIPT OPERATES ON a_N_Grab_for_Assign; BINARIZES IT; FOR EACH CLASS
# IT SEARCHES FOR THE BEST One_vs_All SEPARATING NEURON
time_start_One_vs_All_NC_Assign_Err_Est = datestr(now)


# ERR ESTIMATE PARAMETERS
# -----------------------
Top_n_Est = 1 # SET TO 1 IF WISH TO CALC P_err_Top_n_P AND P_err_Top_n_N

n_Top_n = 5 # DEFINES TOP CLASSES FOR ERR ESTIMATE; USED BY 
#       NC_IMGNT_RSNT152_HUL_1vsALL_NC_Assgn_Top5_Err_Est_SCRIPT_230317
Lambda_Perr = 0.7# USE 0.7 FOR 20 AND 50 CLASSES; NC ASSIGN SEGMENT BEST 
# REGULARIZER FACTOR FOR Perr; BEST VALUE IS RESULT FROM 
# NC_IMAGENET_HUL_1vsALL_NC_Assgn_ANALYZER_040417
Lambda_Pe_Top_n = 0.06# USE 0.07 AS FOR  BEST REGULARIZER FACTOR FOR Pe_Top_n FOR
# (n_Top_n, Nr_Classes) =, (3, 20), 0.08 FOR, (n_Top_n, Nr_Classes) =, (3, 50), AND 0.06 
# FOR (n_Top_n, Nr_Classes) =, (5, 50); 

y_N = zeros(n_N, 1); 
    
wx_N = zeros(n_N, 1); # USEFUL WHEN RUN INDEPENDENTLY

# SEPARATE MATRIX CALC SECTION
# ----------------------------
eval(["load classifier_features_filename Means " ]); # TO BE USED ONLY
#            TO CALC Any_Separate_List ETC FOR GRABBING MEM SAVING PURPOSE

# Separate_Matrix AND RELATED ENTITIES CALCULATION; USED AT 'NEURAL CODE
# GRABBING' BELOW
Separate_Matrix = zeros(Nr_Classes, n_N); # THE (1, 0) ENTRIES OF EACH
# COLUMN REPRESENT WHICH CLASS Means ARE ON 1 HALF SPACE OF A GIVEN
# HP/ NEURONS AND WHICH ARE ON THE OTHER HALF SPACE; WE HAVE AS MANY
# COLUMNS AS NEURONS
for nn = 1 : n_N # ALL NEURONS

    for dd = 1 : Nr_Classes # ALL CLASSES

        Separate_Matrix[dd, nn] = ...
            (sign(transpose(w_N[:, nn]) * Means[:, dd] - theta_N[nn]) + 1) / 2

    end

end

Separate_Nr = sum(Separate_Matrix); # AN n_N ELEMENTS VECTOR; EACH
# ENTRY CONTAINS THE NR OF CLASS Means WHICH RESIDE ON THE +1 HALF
# SPACE OF THIS HP/ NEURON
Any_Separate_HP_Total = sum(abs(Separate_Nr) ~= 0 & Separate_Nr ~= Nr_Classes); # TOTAL NR OF
# ANY-SEPARATING HPs; FOR INSPECTION ONLY
Any_Separate_List = find(abs(Separate_Nr) ~= 0 & Separate_Nr ~= Nr_Classes); # CONTAINS LIST
# OF HP's [INDEXES TO Separate_Matrix COLUMNS] WHICH SEPARATE
# "ANY VS REST' CLASSES Means WHERE 'ANY" IS ANY NON-ZERO
# LIST LENGTH IS Any_Separate_HP_Total
One_Separate_HP_Total = sum(Separate_Nr .== 1); # FOR INFO ONLY; TOTAL NR OF
# ONE-SEPARATING HPs; CLEARLY Any_Separate_HP_Total >= One_Separate_HP_Total
One_Separate_List = find(Separate_Nr .== 1); # FOR INFO ONLY; CONTAINS LIST
# OF HP's [INDEXES TO Separate_Matrix COLUMNS] WHICH SEPARATE
# "1 VS ALL" CLASSES Means; LIST LENGTH IS One_Separate_HP_Total
eval(["save ' multi_epoch_classifier_results_filename ' Separate_Matrix Any_Separate_HP_Total Any_Separate_List  -append"]); #

# NC ASSIGN SEGMENT
# -----------------
a_N_Grab_for_Assign = zeros(Any_Separate_HP_Total, Train_Batch_Size)
# WE NOW LOAD ASSOCIATION BATCH [BY DEFAULT S_1 BATCH], RUN NEURON, 
# ACTIVITY [ONLY] AND GRAB
eval(["load ' classifier_features_filename '  S_1  C_1"])
# BATCH PROCESSING
for tx =  1 : Train_Batch_Size
    # PICK NEW x
    # ----------
    # HUL INPUT FEATURE VECTOR PREPARE    
    eval(["S = S_1[:, tx] "]); # DIM IS d 

    for nn = 1 : n_N # FOR EACH INPUT SAMPLE RUN THRU ALL n_N = nd * d NEURONS
        # NEURAL OUTPUT, SEMI-LINEAR [NO SAT] NEURAL OUTPUT
        # -------------------------------------------------
        [y_N[nn], wx_N[nn]] = HUL_SemiLinearNoSat_Neuron_Activity_070915...
            (S, w_N[:, nn], theta_N[nn], d)

    end # NEXT NEURON nn

    a_N_Grab_for_Assign[:, tx] = wx_N[Any_Separate_List] - theta_N[Any_Separate_List]; # DIM IS n_N AT THIS STAGE

end # NEXT WITHIN BATCH SAMPLE tx

eval(["S_labels = C_1 "])
# WE TAKE NEURAL CODE [ANY_SEPARATE SUBSET, RESULT OF HUL CLASSIFIER]
# TRANSPOSE; SUBSTITUTE ALL NON-NEG BY +1; ALL NEG BY -1
y = 2 * (a_N_Grab_for_Assign .> 0) - 1;# THIS IS A
# Any_Separate_HP_Total * Train_Batch_Size  DIM MATRIX


y = y'; # y NOW IS A Train_Batch_Size * Any_Separate_HP_Total DIM MATRIX

One_vs_All_Sep_Array_Perr = zeros(Nr_Classes, 2); # ENTRIES OF 
# One_vs_All_Sep_Array_Perr[:, :] ARE BEST POS_SEPARATING NEURON NR FOR EACH CLASS
# 2nd INDEX DENOTES BEST NEURON [1] AND BEST NEURON QUALITY [2]
One_vs_All_Sep_Array_Pe_Top_n = zeros(Nr_Classes, 2); # ENTRIES 
# ARE BEST POS_SEPARATING NEURON NR FOR EACH CLASS; FOR Pe_Top_n
# 2nd INDEX DENOTES BEST NEURON [1] AND BEST NEURON QUALITY [2]
samples_I_Assgn = zeros(1, Nr_Classes); # WILL CONTAIN TOTAL NR OF SAMPLES OF EACH CLASS
#                                   IN BATCH
y_column_sums_I = zeros(Nr_Classes, Any_Separate_HP_Total)

samples_O_Assgn = zeros(1, Nr_Classes); # WILL CONTAIN TOTAL NR OF SAMPLES NOT IN
#                                       EACH CLASS IN BATCH
y_column_sums_O = zeros(Nr_Classes, Any_Separate_HP_Total)

Batch_range = 1 : Train_Batch_Size; 

for dd = 1 : Nr_Classes # PRE-CALC

    samples_I_Assgn[dd] = sum(S_labels[Batch_range] .== Selected_Classes[dd]); # TOTAL SAMPLES 
    #  IN BATCH WITHIN SELECTED CLASS [SUFFIX _I DENOTES "WITHIN"]
    indxs_I = find(S_labels[Batch_range] .== Selected_Classes[dd]); # INDEXES VECTOR OF SELECTED
    # CLASS SAMPLES; EACH indxs ELEMENT RANGES 1 : Batch_Range
    y_column_sums_I[dd, :] = sum(y[indxs_I, :]); # A ROW VECTOR OF DIMENSION
    # Any_Separate_HP_Total; EACH ELEMENT IS THE SUM OF THOSE ELEMENTS OF THE
    # CORRESPONDING COLUMN OF y WHICH BELONG TO THE SELECTED CLASS
    samples_O_Assgn[dd] = sum(S_labels[Batch_range] ~= Selected_Classes[dd]); # TOTAL SAMPLES 
    #  IN BATCH NOT OF SELECTED CLASS [SUFFIX _O DENOTES "NOT OF SELECTED CLASS"] 
    indxs_O = find(S_labels[Batch_range] ~= Selected_Classes[dd]); # INDEXES VECTOR OF SAMPLES
    # NOT IN SELECTED CLASS ; EACH indxs ELEMENT RANGES 1 : Batch_Range
    y_column_sums_O[dd, :] = sum(y[indxs_O, :]); # A ROW VECTOR OF DIMENSION
    # Any_Separate_HP_Total; EACH ELEMENT IS THE SUM OF THOSE ELEMENTS OF THE
    # CORRESPONDING COLUMN OF y WHICH DO NOT BELONG TO THE SELECTED CLASS
end

# ONE vs ALL SEP ARRAY; Perr CALC
# -------------------------------
for dd = 1 : Nr_Classes

    best_metric = -inf

    for nn = 1 : Any_Separate_HP_Total

        candidate_metric = y_column_sums_I[dd, nn] -...
           Lambda_Perr * y_column_sums_O[dd, nn]

        if candidate_metric .> best_metric

            best_metric = candidate_metric

            best_neuron = nn; # INDEXES OF y VECTOR [SUBSET OF FULL NC]

        end

    end # NEXT NEURON

    One_vs_All_Sep_Array_Perr[dd, 1] = best_neuron

    One_vs_All_Sep_Array_Perr[dd, 2] = best_metric

end # NEXT dd

# TRIO SEP; POS; NEG POLARITY BEST NEURON AND QUALITY ARRAYS
One_vs_All_Best_Neuron_Perr = One_vs_All_Sep_Array_Perr[:, 1];# A Nr_Classes VECTOR
#                                           MAY DISPLAY IF REQUIRED
One_vs_All_Neuron_Metrics_Perr = One_vs_All_Sep_Array_Perr[:, 2]

# ONE vs ALL SEP ARRAY; Pe_Top_n CALC
# -----------------------------------
for dd = 1 : Nr_Classes

    best_metric = -inf

    for nn = 1 : Any_Separate_HP_Total

        candidate_metric = y_column_sums_I[dd, nn] -...
           Lambda_Pe_Top_n * y_column_sums_O[dd, nn]

        if candidate_metric .> best_metric

            best_metric = candidate_metric

            best_neuron = nn; # INDEXES OF y VECTOR [SUBSET OF FULL NC]

        end

    end # NEXT NEURON

    One_vs_All_Sep_Array_Pe_Top_n[dd, 1] = best_neuron

    One_vs_All_Sep_Array_Pe_Top_n[dd, 2] = best_metric

end # NEXT dd

# TRIO SEP; POS; NEG POLARITY BEST NEURON AND QUALITY ARRAYS
One_vs_All_Best_Neuron_Pe_Top_n = One_vs_All_Sep_Array_Pe_Top_n[:, 1];# A Nr_Classes VECTOR
#                                           MAY DISPLAY IF REQUIRED
One_vs_All_Neuron_Metrics_Pe_Top_n = One_vs_All_Sep_Array_Pe_Top_n[:, 2]


time_complete_One_vs_All_NC_Assign_Err_Est = datestr(now)


# ERROR ESTIMATE SEGMENT
# ----------------------
a_N_Grab_for_Err_Est = zeros(Any_Separate_HP_Total, Test_Batch_Size)
# WE NOW LOAD S_Test BATCH, RUN NEURON ACTIVITY [ONLY] AND GRAB
eval(["load ' classifier_features_filename ' S_Test  C_Test " ])

eval(["S_labels = C_Test "])

# PRE-CALC
Batch_range = 1 : Test_Batch_Size

samples_I = zeros(1, Nr_Classes); # WILL CONTAIN TOTAL NR OF SAMPLES OF EACH CLASS
#                                   IN BATCH
samples_O = zeros(1, Nr_Classes)

for dd = 1 : Nr_Classes

    samples_I[dd] = sum(S_labels[Batch_range] .== Selected_Classes[dd]); # TOTAL SAMPLES
    #  IN BATCH WITHIN SELECTED CLASS [SUFFIX _I DENOTES "WITHIN"]
    samples_O[dd] = sum(S_labels[Batch_range] ~= Selected_Classes[dd]); # TOTAL SAMPLES
    #  IN BATCH NOT OF SELECTED CLASS [SUFFIX _O DENOTES 'NOT OF SELECTED
    #  CLASS']
end
# BATCH PROCESSING
for tx =  1 : Test_Batch_Size
    # PICK NEW x
    # HUL INPUT FEATURE VECTOR PREPARE    
    eval(["S = S_Test[:, tx] "]); #zz A FEATURE VECTOR, DIM IS d
    
    for nn = 1 : n_N # FOR EACH INPUT SAMPLE RUN THRU ALL n_N = nd * d NEURONS

        # NEURAL OUTPUT, SEMI-LINEAR [NO SAT] NEURAL OUTPUT
        # -------------------------------------------------
        [y_N[nn], wx_N[nn]] = HUL_SemiLinearNoSat_Neuron_Activity_070915...
            (S, w_N[:, nn], theta_N[nn], d)

    end # NEXT NEURON nn

    a_N_Grab_for_Err_Est[:, tx] = wx_N[Any_Separate_List] - theta_N[Any_Separate_List]; # DIM IS n_N AT THIS STAGE

    # SOFTMAX ACTIVATION FUNCTION
    a_shift = -max(a_N_Grab_for_Err_Est[:, tx]); # USED TO REDUCE PROBABILITY
    #   OF softmax[.] NaN RESULT; SEE e.g. BENDERSKY'S WEB POST ON SOFTMAX
    a_N_Grab_for_Err_Est[:, tx] = softmax[a_N_Grab_for_Err_Est[:, tx] + a_shift]

end # NEXT WITHIN BATCH SAMPLE tx

# EXECUTES One_vs_All ERROR ESTIMATION.
# Pe [CLASS] AND Perr [TOTAL]IS CALCULATED BY BINARY NEURAL OUTPUT
# MAY BE RUN IN 2 WAYS: 1. AUTOMATICALLY INVOKED BY NC_IMAGENET_HUL_Classifier
#                       2. MANUALLY; WITH PRE-LOAD OF NC_IMAGENET_HUL_Classifier
#                          RESULTS FILE (SUCH AS HUL_Classifier_IMAGENET_Results_020317

# ONE VS ALL POS POLARITY DECISION STAGE; e.g. TEOW_LOE_2002
# ----------------------------------------------------------
One_vs_All_P_err_ctr = zeros(1, Nr_Classes)

One_vs_All_P_Confusion_Matrix = zeros(Nr_Classes, Nr_Classes)

for tt =  1 : Test_Batch_Size
    
    One_vs_All_P_decided_metric = -inf

    for dd = 1 : Nr_Classes        

        candidate_metric = a_N_Grab_for_Err_Est[One_vs_All_Best_Neuron_Perr[dd], tt]

        if candidate_metric .> One_vs_All_P_decided_metric 

            One_vs_All_P_decided_metric = candidate_metric

            One_vs_All_P_dec_dd = dd

        end

    end # NEXT CLASS dd
    # ONE VS ALL POS POLARITY CONFUSION MATRIX AND ERROR COUNT
    One_vs_All_P_Confusion_Matrix...
        (One_vs_All_P_dec_dd, find(S_labels[tt] .== Selected_Classes)) =...
        One_vs_All_P_Confusion_Matrix...
        (One_vs_All_P_dec_dd, find(S_labels[tt] .== Selected_Classes)) + 1

    if find(S_labels[tt] .== Selected_Classes) ~= One_vs_All_P_dec_dd # ERR EVENT

        One_vs_All_P_err_ctr[find(S_labels[tt] .== Selected_Classes)] =...
            One_vs_All_P_err_ctr[find(S_labels[tt] .== Selected_Classes)] + 1

    end

end # NEXT SAMPLE tt

One_vs_All_P_Confusion_Matrix = One_vs_All_P_Confusion_Matrix

Perr_One_vs_All_P = sum(One_vs_All_P_err_ctr) / Test_Batch_Size

Pe_Class_One_vs_All_P = One_vs_All_P_err_ctr ./ samples_I

# Pe_Top_n ERROR ESTIMATE SEGMENT
# -------------------------------
if Top_n_Est .== 1    
    # P_err_Top_n_P CALC
    Soft_P_Out = zeros(1, Nr_Classes)

    Top_n_P_err_ctr = 0

    for tt =  1 : Test_Batch_Size

        for dd = 1 : Nr_Classes

            Soft_P_Out[dd] = a_N_Grab_for_Err_Est[One_vs_All_Best_Neuron_Pe_Top_n[dd], tt]

        end # NEXT CLASS dd

        [Sorted_Soft_P_Out, Sorted_indxs] = sort(Soft_P_Out, "descend")

        if all(S_labels[tt] ~= Selected_Classes[Sorted_indxs[1 : n_Top_n]])#zz

            Top_n_P_err_ctr = Top_n_P_err_ctr + 1

        end

    end # NEXT SAMPLE tt

    P_err_Top_n_P = Top_n_P_err_ctr / Test_Batch_Size    

end# END TOP-n ERR EST

eval(["save ' multi_epoch_classifier_results_filename ' Perr_One_vs_All_P   Pe_Class_One_vs_All_P   One_vs_All_P_Confusion_Matrix   P_err_Top_n_P   Top_n_Est  n_Top_n   Lambda_Perr   Lambda_Pe_Top_n  -append"]); # SAVE 
# RESULTS VARS INCL. One_vs_All_P_Confusion_Matrix ETC
time_finish_HUL = datestr(now)