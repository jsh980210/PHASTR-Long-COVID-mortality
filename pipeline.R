

@transform_pandas(
    Output(rid="ri.vector.main.execute.beb33798-6230-4fd5-9b8a-26761eba873a")
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

