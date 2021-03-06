library(shiny)
library(MASS)

ui <- fluidPage(
  
  titlePanel("Bayes"),
  
  sidebarLayout(
    sidebarPanel(
      fluidRow(
        column(12, radioButtons("classifiers", "Classificator", c("Naive" = 1), selected = 1, inline = T)),
        column(12, sliderInput("n", "amount of observations", 100, 500, 500, 100)),
        
        column(12, "Wolves", style = "color: gray; text-align: center; font-size: 24px"),
        column(6, sliderInput("mu11", "Math.Expectation First", 10, 20, 10, 1)), column(6, sliderInput("mu21", "Math.Expectation Second", 1, 10, 5, 1)),
        column(6, sliderInput("sigma11", "Standard deviation First", 0.1, 1, 0.5, 0.1)), column(6, sliderInput("sigma21", "Standard dev. Second", 0.1, 1, 0.5, 0.1)),
        
        column(12, "Foxes", style = "color: orange; text-align: center; font-size: 24px"),
        column(6, sliderInput("mu12", "Math.Expectation First", 10, 20, 15, 1)), column(6, sliderInput("mu22", "Math.Expectation Second", 1, 10, 5, 1)),
        column(6, sliderInput("sigma12", "Standard deviation First", 0.1, 1, 0.5, 0.1)), column(6, sliderInput("sigma22", "Standard dev. Second", 0.1, 1, 0.5, 0.1))
        
      )
    ),
    
    mainPanel(
      plotOutput("plot"),
      textOutput("error")
    )
  )
)

naive_bayes <- function(xm) {
  p <- function(ksi, mu, sigma) (1 / (sigma * sqrt(2 * pi))) * exp(-(ksi - mu) ^ 2 / (2 * sigma ^ 2))
  
  classifier <- function(x, classes, mu, sigma, Py, lambda = c(1, 1)) {
    sum_by_class <- rep(0, length(classes))
    names(sum_by_class) <- classes
    for (i in 1:length(sum_by_class)) {
      sum <- 0
      for (j in 1:length(x)) sum <- sum + log(p(x[j], mu[i,j], sigma[i,j]))
      sum_by_class[i] <- log(lambda[i] * Py[i]) + sum
    }
    names(which.max(sum_by_class))
  }
  
  build_classification_map <- function(classes, mu, sigma, Py, limits) {
    classifiedObjects <- c()
    for (i in seq(limits[1,1] - 5, limits[1,2] + 5, 0.1))
      for (j in seq(limits[2,1] - 5, limits[2,1] + 5, 0.1)) 
        classifiedObjects <- rbind(classifiedObjects, c(i, j, classifier(c(i, j), classes, mu, sigma, Py)))
      classifiedObjects
  }
  
  draw_plot <- function(xm, classified_objects) {
    n <- ncol(xm)
    colors <- c("gray" = "gray", "orange" = "orange")
    plot(xm[,1:(n-1)], pch = 21, bg = colors[xm[,n]], col = colors[xm[,n]], main = "Classification map of Normal dis.", asp = 1)
    points(classified_objects[,1:(n-1)], pch = 21, col = colors[classified_objects[,n]])
  }
  
  get_mu <- function(xm) sum(xm) / length(xm)
  
  get_sigma <- function(xm, mu) sum((xm - mu)^2) / (length(xm)-1)
  
  main <- function(xm) {
    Py <- c(0.5, 0.5)
    m <- nrow(xm)
    tmp11 <- xm[1:(m/2),1]
    tmp21 <- xm[1:(m/2),2]
    tmp12 <- xm[(m/2+1):m,1]
    tmp22 <- xm[(m/2+1):m,2]
    mu <- rbind(c(get_mu(tmp11), get_mu(tmp21)), c(get_mu(tmp12), get_mu(tmp22)))
    sigma <- rbind(c(get_sigma(tmp11, mu[1,1]), get_sigma(tmp21, mu[1,2])), c(get_sigma(tmp12, mu[2,1]), get_sigma(tmp22, mu[2,2])))
    classes <- unique(xm[,ncol(xm)])
    limits <- matrix(c(min(mu[,1]), min(mu[,2]), max(mu[,1]), max(mu[,2])), 2, 2)
    
    classified_objects <- build_classification_map(classes, mu, sigma, Py, limits)
    draw_plot(xm, classified_objects)
  }
  
  main(xm)
}


server <- function(input, output) {
  
  output$plot <- renderPlot({
    output$error = renderText("")
    n <- input$n
    xm11 <- rnorm(n/2, input$mu11, input$sigma11)
    xm12 <- rnorm(n/2, input$mu21, input$sigma21)
    xm21 <- rnorm(n/2, input$mu12, input$sigma12)
    xm22 <- rnorm(n/2, input$mu22, input$sigma22)
    tmp1 <- cbind(xm11, xm12)
    tmp2 <- cbind(xm21, xm22)
    colnames(tmp1) <- c()
    colnames(tmp2) <- c()
    xm <- data.frame()
    xm <- rbind(xm, tmp1)
    xm <- rbind(xm, tmp2)
    classes <- 1:n
    classes[1:(n/2)] <- "gray"
    classes[(n/2+1):n] <- "orange"
    xm <- cbind(xm, classes)
    colnames(xm) <- c("First", "Second", "Class")
    if (input$classifiers == 0) optimal_bayes(xm, input)
    else if (input$classifiers == 1) naive_bayes(xm)
    else if (input$classifiers == 2) {
      sigma1 <- matrix(c(input$sigma11, 0, 0, input$sigma21), 2, 2)
      sigma2 <- matrix(c(input$sigma12, 0, 0, input$sigma22), 2, 2)
      mu1 <- c(input$mu11, input$mu21)
      mu2 <- c(input$mu12, input$mu22)
      xm1 <- mvrnorm(n = n/2, mu1, sigma1)
      xm2 <- mvrnorm(n = n/2, mu2, sigma2)
      colnames(xm1) <- c()
      colnames(xm2) <- c()
      xm <- data.frame()
      xm <- rbind(xm, xm1)
      xm <- rbind(xm, xm2)
      classes <- 1:n
      classes[1:(n/2)] <- "gray"
      classes[(n/2+1):n] <- "orange"
      xm <- cbind(xm, classes)
      colnames(xm) <- c("First", "Second", "Class")
      plug_in(xm)
    }
    else if (input$classifiers == 3) {
      sigma1 <- matrix(c(input$sigma11, 0, 0, input$sigma21), 2, 2)
      if (sigma1[1,1] != input$sigma12 || sigma1[2,2] != input$sigma22) {
        output$error = renderText("Standard deviation for attributes of classes must be equal")
        return ()
      }
      mu1 <- c(input$mu11, input$mu21)
      mu2 <- c(input$mu12, input$mu22)
      xm1 <- mvrnorm(n = n/2, mu1, sigma1)
      xm2 <- mvrnorm(n = n/2, mu2, sigma1)
      colnames(xm1) <- c()
      colnames(xm2) <- c()
      xm <- data.frame()
      xm <- rbind(xm, xm1)
      xm <- rbind(xm, xm2)
      classes <- 1:n
      classes[1:(n/2)] <- "gray"
      classes[(n/2+1):n] <- "orange"
      xm <- cbind(xm, classes)
      colnames(xm) <- c("First", "Second", "Class")
      fisher(xm)
    }
  })
}

shinyApp(ui = ui, server = server)
