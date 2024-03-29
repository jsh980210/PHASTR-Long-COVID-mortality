

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.b29fdc92-4983-44c5-853f-c3117d55cf86"),
    analysis_1_COVID_negative_control_pre_matching_first_half=Input(rid="ri.foundry.main.dataset.4bc2a605-ffb8-4a08-b9c2-623fa9730224")
)
analysis_1_COVID_negative_control_matching_first_half <- function(analysis_1_COVID_negative_control_pre_matching_first_half) {
    library(MatchIt)
    set.seed(2023)
    # add seed 
    
    df <- analysis_1_COVID_negative_control_pre_matching_first_half
    matching <- matchit(long_covid ~ data_partner_id + age + log_number_of_visits_per_month_before_index_date + index_date_numberofdays_from_20200101, data = df, method = 'nearest', exact = 'data_partner_id', caliper=c(age = 10, index_date_numberofdays_from_20200101 = 45, log_number_of_visits_per_month_before_index_date = 1), std.caliper = c(FALSE, FALSE, FALSE), ratio = 1)
    
    return (match.data(matching))
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.726b85fa-b487-4360-b8ab-36a5b61bf153"),
    analysis_1_COVID_negative_control_pre_matching_second_half=Input(rid="ri.foundry.main.dataset.17e8e712-5129-47f4-b421-b72c8f8a8d83")
)
analysis_1_COVID_negative_control_matching_second_half <- function(analysis_1_COVID_negative_control_pre_matching_second_half) {
    library(MatchIt)
    set.seed(2023)
    # add seed 
    
    df <- analysis_1_COVID_negative_control_pre_matching_second_half
    matching <- matchit(long_covid ~ data_partner_id + age + log_number_of_visits_per_month_before_index_date + index_date_numberofdays_from_20200101, data = df, method = 'nearest', exact = 'data_partner_id', caliper=c(age = 10, index_date_numberofdays_from_20200101 = 45, log_number_of_visits_per_month_before_index_date = 1), std.caliper = c(FALSE, FALSE, FALSE), ratio = 1)
    
    return (match.data(matching))
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.7aa4122a-d05e-4e3a-999a-88e069107fbd"),
    analysis_1_COVID_positive_control_pre_matching=Input(rid="ri.foundry.main.dataset.ed878439-adf1-44c1-b96e-3b45eb3b6a2d")
)
analysis_1_COVID_positive_control_matching <- function(analysis_1_COVID_positive_control_pre_matching) {
    library(MatchIt)
    set.seed(2023)
    # add seed 
    df <- analysis_1_COVID_positive_control_pre_matching
    matching <- matchit(long_covid ~ data_partner_id + age_at_covid + log_number_of_visits_per_month_before_index_date + index_date_numberofdays_from_20200101, data = df, method = 'nearest', exact = 'data_partner_id', caliper=c(age_at_covid = 10, index_date_numberofdays_from_20200101 = 45, log_number_of_visits_per_month_before_index_date = 1), std.caliper = c(FALSE, FALSE, FALSE), ratio = 1)
    return (match.data(matching))
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.5be15385-d4c0-4a6a-ba59-c12b29c0541e"),
    analysis_1_cohort=Input(rid="ri.foundry.main.dataset.cd475047-2ef9-415c-8812-8336515c5c1f")
)
analysis_1_logistic <- function(analysis_1_cohort) {
    library(broom)
    # seed 
    set.seed(2023)
    df <- analysis_1_cohort
    #df$subcohort <- as.factor(df$subcohort)
    #df$number_of_COVID_vaccine_doses <- as.factor(df$number_of_COVID_vaccine_doses)

    lr <- glm(death ~ PASC + COVID_positive_control + COVID_negative_control + number_of_COVID_vaccine_doses + CCI, data = df, family = binomial)

    print(summary(lr))
    print(exp(coefficients(lr)))
    mod_tbl <- broom::tidy(lr, conf.int = TRUE, exponentiate = TRUE)

    return (mod_tbl)
    # grid search on the threshold to max the recall
    
}

@transform_pandas(
    Output(rid="ri.vector.main.execute.bbf99a67-8010-4c4b-89ea-d35155ea1f46"),
    analysis_1_cohort=Input(rid="ri.foundry.main.dataset.cd475047-2ef9-415c-8812-8336515c5c1f")
)
analysis_1_logistic_MO <- function(analysis_1_cohort) {
    library(broom)
    # seed 
    set.seed(2023)
    df <- analysis_1_cohort
    #df$subcohort <- as.factor(df$subcohort)
    #df$number_of_COVID_vaccine_doses <- as.factor(df$number_of_COVID_vaccine_doses)

    lr <- glm(death ~ morbid_obesity, data = df, family = binomial)

    print(summary(lr))
    print(exp(coefficients(lr)))
    mod_tbl <- broom::tidy(lr, conf.int = TRUE, exponentiate = TRUE)

    return (mod_tbl)
    # grid search on the threshold to max the recall
    
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.daf8e1f1-735d-49eb-a2ef-c864cb7fb0d9"),
    analysis_2a=Input(rid="ri.foundry.main.dataset.dfd52b0d-1b4b-49d1-a420-0f3df44e0f8d")
)
analysis_2a_logistic <- function(analysis_2a) {
    library(broom)
    # seed 
    set.seed(2023)
    df <- analysis_2a
    #df$subcohort <- as.factor(df$subcohort)
    #df$number_of_COVID_vaccine_doses <- as.factor(df$number_of_COVID_vaccine_doses)

    lr <- glm(COVID_patient_death_indicator ~ ., data = df, family = binomial)

    print(summary(lr))
    print(exp(coefficients(lr)))
    mod_tbl <- broom::tidy(lr, conf.int = TRUE, exponentiate = TRUE)

    return (mod_tbl)
    # grid search on the threshold to max the recall
    
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.2a95bda9-68c8-4669-ae3f-977376ed6e98"),
    analysis_2b=Input(rid="ri.foundry.main.dataset.f251c730-78fb-4044-8c57-96c16e3c2011")
)
analysis_2b_logistic <- function(analysis_2b) {
    library(broom)
    # seed 
    set.seed(2023)
    df <- analysis_2b
    #df$subcohort <- as.factor(df$subcohort)
    #df$number_of_COVID_vaccine_doses <- as.factor(df$number_of_COVID_vaccine_doses)

    lr <- glm(COVID_patient_death_indicator ~ ., data = df, family = binomial)

    print(summary(lr))
    print(exp(coefficients(lr)))
    mod_tbl <- broom::tidy(lr, conf.int = TRUE, exponentiate = TRUE)

    return (mod_tbl)
    # grid search on the threshold to max the recall
    
}

@transform_pandas(
    Output(rid="ri.vector.main.execute.beb33798-6230-4fd5-9b8a-26761eba873a"),
    analysis_1_logistic=Input(rid="ri.foundry.main.dataset.5be15385-d4c0-4a6a-ba59-c12b29c0541e")
)
plot_odds_ratio <- function(analysis_1_logistic) {
    library(tidyverse)
    library(ggplot2)
    df <- analysis_1_logistic
    df <- df_clean_function(df)
    
    par(mfrow = c(1, 1))

    # Creates color scheme for p value significance
    SigCols <- c('Non-Significant'="skyblue2", 'Significant'="orange2")
    
    forest_plot1 = ggplot(data=df, aes(y=term, x=estimate, xmin=conf_low, xmax=conf_high, color=Pval_Signif)) + geom_point(size=3) + geom_errorbarh(size=0.85,height=.3) + scale_color_manual(values=SigCols) + labs(title='Outcome: patient death', x='Odds Ratio (OR)', y = 'Variable', color='p-Value Significance') + geom_vline(xintercept=1, color='black', linetype='dashed', alpha=.5) + theme_classic()+theme(text = element_text(size = 20)) + theme(axis.text.x= element_text(size=18))+theme(axis.text.y = element_text(size=18))
    plot(forest_plot1)
    

    
    return(df)
    
}

df_clean_function <- function(df){

    df <- df %>%
        select(-statistic, -std_error) %>%
        mutate('Pval_Signif'=ifelse(p_value<0.05, "Significant", "Non-Significant"))
        
    
    return(df)

}

@transform_pandas(
    Output(rid="ri.vector.main.execute.ed814114-7489-40b6-bf5b-647053bf2d41"),
    analysis_2a_logistic=Input(rid="ri.foundry.main.dataset.daf8e1f1-735d-49eb-a2ef-c864cb7fb0d9")
)
plot_odds_ratio_analysis_2a <- function(analysis_2a_logistic) {
    library(tidyverse)
    library(ggplot2)
    df <- analysis_2a_logistic
    df <- df[df$conf_high <= 10, ]
    df <- df_clean_function(df)
    
    par(mfrow = c(1, 1))

    # Creates color scheme for p value significance
    SigCols <- c('Non-Significant'="skyblue2", 'Significant'="orange2")
    
    forest_plot1 = ggplot(data=df, aes(y=term, x=estimate, xmin=conf_low, xmax=conf_high, color=Pval_Signif)) + geom_point(size=3) + geom_errorbarh(size=0.85,height=.3) + scale_color_manual(values=SigCols) + labs(title='Outcome: patient death', x='Odds Ratio (OR)', y = 'Variable', color='p-Value Significance') + geom_vline(xintercept=1, color='black', linetype='dashed', alpha=.5) + theme_classic()+theme(text = element_text(size = 15)) + theme(axis.text.x= element_text(size=10))+theme(axis.text.y = element_text(size=10))
    plot(forest_plot1)
    

    
    return(df)
    
}

df_clean_function <- function(df){

    df <- df %>%
        select(-statistic, -std_error) %>%
        mutate('Pval_Signif'=ifelse(p_value<0.05, "Significant", "Non-Significant"))
        
    
    return(df)

}

@transform_pandas(
    Output(rid="ri.vector.main.execute.d21dcbb1-51b2-4ba7-9652-07776444b9fa"),
    analysis_2b_logistic=Input(rid="ri.foundry.main.dataset.2a95bda9-68c8-4669-ae3f-977376ed6e98")
)
plot_odds_ratio_analysis_2b <- function(analysis_2b_logistic) {
    library(tidyverse)
    library(ggplot2)
    df <- analysis_2b_logistic
    df <- df_clean_function(df)
    
    par(mfrow = c(1, 1))

    # Creates color scheme for p value significance
    SigCols <- c('Non-Significant'="skyblue2", 'Significant'="orange2")
    
    forest_plot1 = ggplot(data=df, aes(y=term, x=estimate, xmin=conf_low, xmax=conf_high, color=Pval_Signif)) + geom_point(size=3) + geom_errorbarh(size=0.85,height=.3) + scale_color_manual(values=SigCols) + labs(title='Outcome: patient death', x='Odds Ratio (OR)', y = 'Variable', color='p-Value Significance') + geom_vline(xintercept=1, color='black', linetype='dashed', alpha=.5) + theme_classic()+theme(text = element_text(size = 15)) + theme(axis.text.x= element_text(size=10))+theme(axis.text.y = element_text(size=10))
    plot(forest_plot1)
    

    
    return(df)
    
}

df_clean_function <- function(df){

    df <- df %>%
        select(-statistic, -std_error) %>%
        mutate('Pval_Signif'=ifelse(p_value<0.05, "Significant", "Non-Significant"))
        
    
    return(df)

}

