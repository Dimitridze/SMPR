#install.packages("plotrix")
require("plotrix")
euclideanDistance <- function(u, v)
{
  return (sqrt(sum((u - v)^2)))
}

kernelE = function(r){  return ((3/4*(1-r^2)*(abs(r)<=1))) }
kernelQ = function(r){  return ((15 / 16) * (1 - r ^ 2) ^ 2 * (abs(r) <= 1)) }
kernelT = function(r){  return ((1 - abs(r)) * (abs(r) <= 1)) }
kernelG = function(r){  return (((2*pi)^(-1/2)) * exp(-1/2*r^2)) }
kernelR = function(r){  return ((0.5 * (abs(r) <= 1) )) } 


getPotentials <- function(XL,eps,h,class) 
{
  # get pots all elements
  l <- nrow(XL)
  n <- ncol(XL)

  potentials <- rep(0,l)
  err <- eps + 1
 
   while (err > eps) 
  {
    while (TRUE) 
     {
      # Ïîêà íå ïîëó÷èì íåñîîòâåòñòâèå êëàññîâ, ÷òîáû îáíîâèòü ïîòåíöèàëû
      rand <- sample(1:l, 1)
x=XL[rand,1:2]
      u <- PF(potentials,XL,x,h)

      if(colors[u] != colors[class[rand]]) {
        potentials[rand] = potentials[rand] + 1
        break
        }
     }
    # Ïîäñ÷åò ÷èñëà îøèáîê
    err <- 0
    for (i in 1:l)
    {
x = XL[i,1:2]
        points=XL[-i,1:3]
         if(colors[PF(potentials,points,x,h)]!= colors[class[i]])
    {
          err = err + 1
    }
  }
}
  return (potentials)
}



PF = function(potentials,XL,y,h,metricFunction = euclideanDistance)
{
   l <- nrow(XL)
   n <- ncol(XL)

  weights = rep(0,3)
  names(weights) = unique(XL[,3])
  for(i in 1:l)
  {
    
    x=XL[i,1:2]
    class=XL[i,3]
    
    r = metricFunction(x,y)/h
    weights[class] = weights[class] + potentials[i]*kernelR(r);
  }
class = names(which.max(weights))
  #no p in w
 if (max(weights) == 0) return("0") 
    return(class)
         }
	



PFpic = function(XL, classes, potentials, h, colors) {
  entialsplot(XL, bg = colors[classes], pch = 21, asp = 1,  main = "Pont-s") 
  t = potentials / max(potentials)
  for (i in 1:l) 
{
      x = XL[i, 1]
      y = XL[i, 2]
      if(t[i]!=0)
{
      color = adjustcolor(colors[classes[i]], t[i] /3)
      rc <- 35 
      draw.circle(x, y, h, rc, border = color, col = color)
      }
  }

  for (i in 1:n) {
    x = XL[i, 1]
    y = XL[i, 2]
    if(t[i]!=0){
  text(x, y, labels = potentials[i], pos=4, col = "black")
    }
  }
}

help(plot)



colors = c("setosa" = "red", "versicolor" = "green", "virginica" = "blue", "0" = "NA")
xl = iris[, 3:5] 
class = iris[, 5]
l <- nrow(xl)
Y = rep(0,l)
text = paste("Map classificaton for (R kernel) with h = ", h=1)
poten = getPotentials(xl,15,h=1,class)
PFpic(xl[,1:2], class, poten, h=1, colors)
