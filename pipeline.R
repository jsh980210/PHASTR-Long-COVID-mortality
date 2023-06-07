

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.b29fdc92-4983-44c5-853f-c3117d55cf86"),
    analysis_1_COVID_negative_control_pre_matching=Input(rid="ri.foundry.main.dataset.4bc2a605-ffb8-4a08-b9c2-623fa9730224")
)
analysis_1_COVID_negative_control_matching <- function(analysis_1_COVID_negative_control_pre_matching) {
    library(MatchIt)
    set.seed(2023)
    # add seed 
    
    df <- analysis_1_COVID_negative_control_pre_matching
    matching <- matchit(long_covid ~ data_partner_id + age + observation_period + index_date_numberofdays_from_20200101, data = df, method = 'nearest', exact = 'data_partner_id', caliper=c(age = 10, index_date_numberofdays_from_20200101 = 45, observation_period = 60), std.caliper = c(FALSE, FALSE, FALSE), ratio = 1)
    
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
    matching <- matchit(long_covid ~ data_partner_id + age_at_covid + observation_period_post_covid + index_date_numberofdays_from_20200101, data = df, method = 'nearest', exact = 'data_partner_id', caliper=c(age_at_covid = 10, index_date_numberofdays_from_20200101 = 45, observation_period_post_covid = 60), std.caliper = c(FALSE, FALSE, FALSE), ratio = 1)
    return (match.data(matching))
}

