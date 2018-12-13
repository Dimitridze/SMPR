# SMPR
Language R

- [Метрические алгоритмы классификации](#Метрические-алгоритмы-классификации)
  - [1NN](#1NN)
  - [KNN](#KNN)
  - [KWNN](#KWNN)
  - [Парзеновское окно (PW)](#Парзеновское-окно-pw)
  - [Потенциальные функции (PF)](#Потенциальные-функции-pf)
  - [Алгоритм STOLP](#Алгоритм-STOLP)
- [Байесовские классификаторы](#Байесовские-классификаторы)
  - [Изолинии](#Изолинии)
  
 
  
  
  <center>
<table>
  <tbody>
    <tr>
      <th>Метод</th>
      <th>Параметры</th>
      <th>Loo</th>
    </tr>
    <tr>
      <td>KWNN</a></td>
      <td>k=9</td>
      <td>0.0333</td>
    </tr>
    <tr>
      <td>KNN</a></td>
      <td>k=6</td>
      <td>0.0333</td>
    </tr>
    <tr>
      <td>Парзеновские окна</a></td>
      <td>h=0.4  ( Прямоугольное ядро )</td>
      <td>0.04</td>
    </tr>
    <tr>
      <td>Парзеновские окна</a></td>
      <td>h=0.4  ( Треугольное ядро )</td>
      <td>0.04</td>
    </tr>
    <tr>
      <td>Парзеновские окна</a></td>
      <td>h=0.4  ( ядро Епанечникова )</td>
      <td>0.04</td>
    </tr>
    <tr>
      <td>Парзеновские окна</a></td>
      <td>h=0.4  ( Квартическое ядро )</td>
      <td>0.04</td>
    </tr>
      <tr>
      <td>Парзеновские окна</a></td>
      <td>h=0.1  ( Гауссовское ядро )</td>
      <td>0.04</td>
    </tr>
    <tr>
      <td>Потенциальные функции</a></td>
      <td>h=0.4  ( Прямоугольное ядро )</td>
      <td>Подбор гамма</td>
    </tr>
       	  </tbody>
   </table>
 
 
## Метрические алгоритмы классификации

**Метрические методы обучения** -- методы, основанные на анализе сходства объектов.

Метрические алгоритмы классификации опираются на **_гипотезу компактности_**: схожим объектам соответствуют схожие ответы.

Метрические алгоритмы классификации с обучающей выборкой *Xl* относят объект *u* к тому классу *y*, для которого **суммарный вес ближайших обучающих объектов ![](https://latex.codecogs.com/gif.latex?W_y%28u%2C%20X%5El%29) максимален**:

![](https://latex.codecogs.com/gif.latex?W_y%28u%2C%20X%5El%29%20%3D%20%5Csum_%7Bi%20%3A%20y_%7Bu%7D%5E%7B%28i%29%7D%20%3D%20y%7D%20w%28i%2C%20u%29%20%5Crightarrow%20max)

где весовая функция *w(i, u)* оценивает степень важности *i*-го соседа для классификации объекта *u*.

Функция ![](https://latex.codecogs.com/gif.latex?W_y%28u%2C%20X%5El%29) называется **_оценкой близости объекта u к классу y_**. Выбирая различную весовую функцию *w(i, u)* можно получать различные метрические классификаторы.

### Методы ближайших соседей

Классификация ирисов Фишера методом 1NN (ближайшего соседа)

Решалась задача классификации. В качестве обучающей выборки была взята матрица ирисов фишера по длине и ширине лепестка.
Требовалось построить карту классификации на основе данных обучающей выборки.
В качестве метода классификации использовался метод 1NN.
## 1NN 
# Метод 1NN состоит в следующем: 
	1.Для классифицируемого объекта вычисляются расстояния от него до каждого объекта обучающей выборки.
	2.Обучающая выборка сортируется по возрастанию расстояния от каждого объекта выборки до классифицируемого
	3.Классифицируемому объекту присваивается тот же класс, что и ближайшего к нему объекта выборки.
 
 
  <p><img src="img\1nn.png" ></p>
  
Классификация ирисов Фишера методом kNN (k ближайших соседей)


Решалась задача классификации. В качестве обучающей выборки была взята матрица ирисов фишера по длине и ширине лепестка.
Требовалось построить карту классификации на основе данных обучающей выборки.
В качестве метода классификации использовался метод kNN.


  ## KNN 


 Метод kNN состоит в следующем: 
	1.Для классифицируемого объекта вычисляются расстояния от него до каждого объекта обучающей выборки
	2.Обучающая выборка сортируется по возрастанию расстояния от каждого объекта выборки до классифицируемого
	3.Подсчитывается, какой класс доминирует среди k ближайших соседей, и этот класс присваивается классифицируемому объекту
	## Алгоритм k ближайших соседей (kNN)
Алгоритм 1NN относит классифицируемый объект U к тому классу, которому принадлежит его ближайший сосед.
ὠ(i,u)=[i=1];

Алгоритм kNN относит объект к тому классу, элементов которого больше среди k ближайших соседей x(i), i=1,..,k.

Для оценки близости классифицируемого объекта *u* к классу *y* **алгоритм kNN** использует следующую функцию:

ὠ(i,u)=[i<=k], где *i* -- порядок соседа по расстоянию к классифицируемому объекту *u*, k-количество параметровю=.
**Реализация весовой функции производится следующим образом**:

``` R
distances <- matrix(NA, l, 2) # расстояния от классифицируемого объекта u до каждого i-го соседа 
for(i in 1:l) {
   distances[i, ] <- c(i, eDist(xl[i, 1:n], u))
}
orderedxl <- xl[order(distances[ , 2]), ] # сортировка расстояний
classes <- orderedxl[1:k, n + 1] # названия первых k классов (k ближайших соседей) в classes 
```

<p><img src="img\LooKnn.png" ></p>

### Преимущества:
1. Простота реализации.
2. При *k*, подобранном около оптимального, алгоритм "неплохо" классифицирует.

### Недостатки:
1. Нужно хранить всю выборку.
2. При *k = 1* неустойчивость к погрешностям (*выбросам* -- объектам, которые окружены объектами чужого класса), вследствие чего этот выброс классифицировался неверно и окружающие его объекты, для которого он окажется ближайшим, тоже.
2. При *k = l* алгоритм наоборот чрезмерно устойчив и вырождается в константу.
3. Крайне бедный набор параметров.
4. Точки, расстояние между которыми одинаково, не все будут учитываться.

## KWNN
Реализаця метода kwNN
В каждом классе выбирается
__k__ ближайших к __U__ объектов, и объект u относится к тому классу, для
которого среднее расстояние до __k__ ближайших соседей минимально.
![](http://latex.codecogs.com/gif.latex?w%28i%2Cu%29%3D%5Bi%5Cleq%20k%5Dw%28i%29)
где,
![](http://latex.codecogs.com/gif.latex?w%28i%29) — строго убывающая последовательность вещественных весов, задающая
вклад i-го соседа при классификации объекта u.

<p><img src="img\LooKwnn.png" ></p>

### Весовая функция

```
weightsKWNN = function(i, k)
{
  (k + 1 - i) / k
}
```

<center>
<table>
  <tbody>
    <tr>
      <th>Метод</th>
      <th>Параметры</th>
      <th>Точность</th>
    </tr>
    <tr>
      <td>KWNN</a></td>
      <td>k=9</td>
      <td>0.0333</td>
    </tr>
    <tr>
      <td>KNN</a></td>
      <td>k=6</td>
      <td>0.0333</td>
    </tr>
	  </tbody>
   </table>

#### Сравнение качества алгоритмов kNN и kwNN.

kNN — один из простейших алгоритмов классификации, поэтому на реальных задачах он зачастую оказывается неэффективным. Помимо точности классификации, проблемой этого классификатора является скорость классификации: если в обучающей выборке N объектов, в тестовой выборе M объектов, и размерность пространства  K, то количество операций для классификации тестовой выборки может быть оценено как O(KMN).

kwNN отличается от kNN, тем что учитывает порядок соседей классифицируемого объекта, улчшая качество классификации.

Пример, показывающий преимущество метода kwNN над kNN:
<a href="http://www.picshare.ru/view/9312480/" target="_blank"><img src="http://www.picshare.ru/uploads/181018/l7iAu3dDsZ.jpg" border="0" width="1462" height="524" title="Хостинг картинок PicShare.ru"></a>

В примере передаем параметр k=5.  Kwnn в отличии от Knn оценивает не только индекс соседа, но и  расстояние до него, из-за этого результат получается более точный, что и продемонстрировано в данном примере.


Число k выбирается методом LOO (скользящего контроля)


### Метод LOO:
1. Элемент удаляется из выборки
2. Запускается алгоритм классификации (в данном случае kNN) для полученной выборки и удалённого объекта
3. Полученный класс объекта сравнивается с реальным. В случае несовпадения классов, величина ошибки увеличивается на 1
4. Процесс повторяется для каждого объекта выборки
5. Полученная ошибка усредняется
6. Процесс повторяется для других значений k. В итоге выбирается число k с наименьшей ошибкой LOO
    

```	
##Составляем таблицу встречаемости каждого класса
counts <- table(classes)
class <- names(which.max(counts))
return (class)
}

## Метод скользящего контроля
Loo <- function(k,xl)
   {
    sum =0
    for(i in 1:dim(xl)[1]
       {
        tmpXL <- rbind(xl[1:i-1, ],
        xl[i+1:dim(xl)[1],])
        xi <- c(xl[i,1], xl[i,2])
        class <-kNN(tmpXL,xi,k)
        if(class != xl[i,3])
        sum=sum+1
       }
   sum=sum/dim(xl)[1]
   return(sum)
  }
```

 ### Преимущества:
 Преимущество LOO состоит в том, что каждый объект ровно один раз участвует в контроле, а длина обучающих подвыборок лишь на единицу меньше длины полной выборки.
 ### Недостатки:
  Недостатком LOO является большая ресурсоёмкость, так как обучаться приходится L раз. Некоторые методы обучения позволяют достаточно быстро перенастраивать внутренние параметры алгоритма при замене одного обучающего объекта другим. В этих случаях вычисление LOO удаётся заметно ускорить. 
  
  
  
  ### Парзеновское окно (PW)

Для оценки близости объекта _u_ к классу _y_ алгоритм использует следующую
функцию:

![](http://latex.codecogs.com/svg.latex?%5Clarge%20W%28i%2C%20u%29%20%3D%20K%28%5Cfrac%7B%5Crho%28u%2C%20x%5Ei_u%29%7D%7Bh%7D%29)
, где 
![](http://latex.codecogs.com/svg.latex?%5Clarge%20K%28z%29) — функция ядра.
  
  Наиболее часто используются следующие типы ядер
  
  <p><img src="img\Kernels.png" ></p>
  
 Код для реализации данных типов ядер:
```
kernelE = function(r){ return ((3/4*(1-r^2)*(abs(r)<=1)))}  #епанечникова
kernelQ = function(r){ return ((15 / 16) * (1 - r ^ 2) ^ 2 * (abs(r) <= 1))}  #квартическое
kernelT = function(r){ return ((1 - abs(r)) * (abs(r) <= 1)) }  #треугольное
kernelG = function(r){ return (((2*pi)^(-1/2)) * exp(-1/2*r^2))}  #гауссовское
kernelR = function(r){ return ((0.5 * (abs(r) <= 1) ))}  #прямоугольное
```
  __Алгоритм__ вокруг нашей классифицируемой точки _u_ строит окружность с радиусом _h_. Далее убираем точки, которые не вошли в окружность. Затем для оставшихся, считаем _weights_, суммируем  по class, и с помощью _names(which.max(weights))_ возвращаем название класса "победителя".
  
  Код алгоритма:
```
PW = function(XL,y,h,metricFunction = euclideanDistance){  
l <- dim(xl)[1]
  weights = rep(0,3)
  names(weights) = unique(xl[,3])
  for(i in 1:l)
  {
        x=XL[i,1:2]
    class=XL[i,3]
        r = metricFunction(x,y)/h
    weights[class]=kernelR(r)+weights[class];
  }
  #no p in w
     if(max(weights)==0){ return ("0") }
     else{ return (names(which.max(weights))) }
                                                        }
```
##   Карта классификации для ядра Епанечникова  и   для Треугольного ядра

<p><img src="img\ET.png" ></p>

##   Карта классификации для Квартического ядра и  для Прямоугольного ядра

<p><img src="img\KR.png" ></p>

##   Карта классификации для Гауссовского ядра


В отличии от предыдущих ядер, Гауссовское ядро устраняет проблему с классификацией точек, не попавших ни в одно окно.

<p><img src="img\G.png" ></p>


##   Loo для ядра Епанечникова   и    для Треугольного ядра

<p><img src="img\ETloo.png" ></p>

##   Loo для Квартического ядра   и   для Прямоугольного ядра

<p><img src="img\QR.png" ></p>

##   Loo для Гауссовского ядра

<p><img src="img\Gloo.png" ></p>

__Наблюдения при реализации__:
    Если _h_ сделать слишком маленьким, классифицуруется заметно меньшее количество точек. Если  же _h_ сделать слишком большим, то  учитываются точки, находящиеся на очень большом расстоянии.
   
__Плюсы:__
- прост в реализации
- хорошее качество классификации при правильно подобраном _h_
- все точки с одинаковым расстоянием будут учитаны

__Минусы:__
- необходимо хранить всю выборку целиком
- небольшой набор параметров
- диапазон параметра _h_ необходимо подбирать самостоятельно(в зависимости от плотности расположения т.)
- если ни одна точка не попала в радиус _h_, алгоритм не способен ее
классифицировать (кроме гауссовского ядра)

## Потенциальные функции (PF)

__Метод потенциальных функций__ - относится к метрическим классификаторам. В отличии от метода парзеновсуих окон, окно строится вокруг обучающих точек.
При классификации объект проверяется на близость к объектам из обучающей выборки. Простыми словами, объекты из обучающей выборки «заряжены» своим классом, а вес каждого из них при классификации зависит от «заряда» и расстояния до классифицируемого объекта.

__Алгоритм метода PF__

   - Изначально для каждого объекта выборки задаём *ширину окна* ![h_i](http://latex.codecogs.com/gif.latex?h_%7Bi%7D) эмпирически (выбирается из собственных соображений).
   - Затем для обучающих объектов вычисляем *силу потенциала* ![gamma_i](http://latex.codecogs.com/gif.latex?%5Cgamma_%7Bi%7D). 
   - После чего каждому объекту выборки присваивается *вес* по формуле ![](http://latex.codecogs.com/gif.latex?w%28x_%7Bi%7D%2Cz%29%3D%5Cgamma_%7Bi%7DK%28%5Cfrac%7B%5Crho%28x%7Bi%7D%2Cz%29%7D%7Bh_%7Bi%7D%7D%29%3D%5Cgamma_%7Bi%7DK%28r%29), ![](http://latex.codecogs.com/gif.latex?K%28r%29) - функция ядра.
   - Суммируем веса объектов одинаковых классов. Класс "победитель" присваивается точке.


    
 __Код метода потенциальных функций:__
 
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
	 
Алгоритм подбора ![gamma_i](http://latex.codecogs.com/gif.latex?%5Cgamma_%7Bi%7D):

   - Множество потенциалов ![gamma_](http://latex.codecogs.com/gif.latex?\gamma_) зануляется. Задается максимально допустимое число ошибок(eps).
   - Из обучающей выборки выбирается очередной объект ![x_i](http://latex.codecogs.com/gif.latex?x_%7Bi%7D).
   - Затем Для ![x_i](http://latex.codecogs.com/gif.latex?x_%7Bi%7D) запускаю алгоритм классификации.
   - Если  полученный класс не совпал с реальным, то *сила потенциала* для выбранного объекта увеличивается на 1. Иначе снова выбирается объект и классифицируется.
   - Алгоритм классификации с полученными значениями потенциалов запускается для каждого объекта выборки. Подсчитывается число ошибок.
   - Если число ошибок меньше заданного, то алгоритм завершает работу. Иначе снова выбирается объект из выборки.
	 
__Код алгоритма подбора ![gamma_](http://latex.codecogs.com/gif.latex?\gamma_):__

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
      # Пока не получим несоответствие классов, чтобы обновить потенциалы
      rand <- sample(1:l, 1)
     x=XL[rand,1:2]
        u <- PF(potentials,XL,x,h)

      if(colors[u] != colors[class[rand]]) {
        potentials[rand] = potentials[rand] + 1
        break
        }
     }
    # Подсчет числа ошибок
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
    
В данной программе использовал прямоугольное ядро. Алгоритм подбирает только силу потенциала , радиус потенциалов h задаются эмпирически(по собственным соображениям).

Список ненулевых потенциалов(15 номеров)

     [1] 3
     [1] 66
     [1] 71
     [1] 78
     [1] 84
     [1] 87
     [1] 94
     [1] 102
     [1] 105
     [1] 107
     [1] 119
     [1] 120
     [1] 122
     [1] 134
     [1] 139


__Задать значение ширины окна для каждого класса__
  
     SvoiH <- function(xl) 
     {
     l <- nrow(xl)
     h <- rep(0, l)
     for(i in 1:l) {
     if (xl[i, ncol(xl)] == "setosa") h[i] <- 1
     else h[i] <- 0.4
                  }
     return (h)
      }
      h <- SvoiH(xl)
      
Подбор потенциалов происходит до тех пор, пока точность алгоритма, не будет меньше заданной ошибки eps.
Иллюстрация результата работы алгоритма:
<p><img src="img\PFF2.png" ></p>

__Преимущества метода потенциальных функций:__

- Большое количество параметров для подбора. 
- Возможность использования не всей выборки

__Недостатки метода:__

- Метод непрост для понимания и алгоритмической реализации;
- Неопределённое время работы алгоритма подбора(если взять маленькое eps) ![gamma_i](http://latex.codecogs.com/gif.latex?%5Cgamma_%7Bi%7D);
- Параметры ![gamma_i](http://latex.codecogs.com/gif.latex?%5Cgamma_%7Bi%7D). и _h_ настраиваются слишком грубо;




### Алгоритм STOLP
Алгоритм СТОЛП (STOLP) — алгоритм отбора эталонных объектов для метрического классификатора.
Отступ - величина, показывающая, насколько объект является типичным представителем своего класса. Отступ равен разности между степенью близости объекта к своему классу и суммой близостей объекта ко всем остальным классам.  
![](http://latex.codecogs.com/gif.latex?M%28x_%7Bi%7D%29%3DW_%7By_%7Bi%7D%7D%28x_%7Bi%7D%2CX%5E%7Bl%7D%29-%5Cmax%20W_%7By%7D%28x_%7Bi%7D%2C%20X%5E%7Bl%7D%29)  
Все объекты обучающей выборки можно разделить на несколько типов:

   - Эталонные объекты - наиболее типичные представители своего класса
   - Неинформативные - объекты, не влияющие значительным образом на качество классификации
   - Пограничные - объекты, имеющие отступ, близкий к нулю. Незначительное изменение в выборке, алгоритме, метрике и т.п. может повлиять на их классификацию.
   - Ошибочные - объекты с отрицательными отступами, классифицируемые неверно.
   - Шумовые объекты (выбросы) - малая группа объектов с большими отрицательными отступами. Их удаление улучшает качество классификации.

Идея алгоритма STOLP состоит в том, чтобы уменьшить исходную выборку, выбрав из неё эталонные объекты. Такой подход уменьшит размер выборки, и может улучшить качество классификации.  На вход подаётся выборка, допустимый порог ошибок и порог фильтрации выбросов. 

__Алгоритм:__ 

   - Удаляем из выборки все выбросы (объекты, отступ которых меньше порога фильтрации выбросов).
   - Затем пересчитав все отступы заново, берем по 1 эталону из каждого класса , и добавляем их в множество эталонов.
   - Классифицируем объекты обучающей выборки, с XL -  множеством эталонов. Считаем число ошибок.
   - Если число ошибок меньше заданного порога, то алгоритм завершается, в противном случае присоединяем ко множеству эталонов точку с наименьшим отступом.


В качестве алгоритма классификации буду использовать(наверное) kwNN(k = 10, q = 0.1)  

## Байесовские классификаторы

<center>
<table>
  <tbody>
    <tr>
      <th>Задание</th>
      <th>Shiny</th>
    </tr>
    <tr>
      <td><a href="#Изолинии">Изолинии(Линии уровня)</a></td>
      <td><a href="https://dimitridze.shinyapps.io/line/">+</a></td>


Байесовский подход опирается на теорему о том, что если плотности распределения классов известны, то алгоритм классификации,  можно выписать в явном виде(если алгоритм имеет минимальную вероятность ошибок). Для оценивания плотностей классов по выборке применяются различные подходы. В этом курсе лекций рассматриваем три: непараметрический, параметрический и оценивание смесей распределений.  

При решении задачи __классификации__, необходимо по известному вектору признаков __x__, определить класс  __y__ к которому принадлежит объект __a(x)__  по формуле : __a(x) = argmax P(y|x)__, для которого при условии x, вероятность класса y - наиболее высока. 


В случае когда __P(y__) одинаковы для всех классов, то следует выбрать класс __у__, у которого в точке __x__, плотность больше.

## Формула Байеса
<img src="https://latex.codecogs.com/gif.latex?p(y|x)=\frac{p(x,y)}{p(x)}&space;=&space;\frac{p(x|y)p(y)}{p(x)}" title="p(y|x)=\frac{p(x,y)}{p(x)} = \frac{p(x|y)p(y)}{p(x)}" /></a>
1. <img src="https://latex.codecogs.com/gif.latex?p(y|x)"/></a> - Апостериорная вероятности, т.е. вероятность того, что объект x принадлежит классу y.
2. <img src="https://latex.codecogs.com/gif.latex?p(x|y)"/></a> - функция правдободобия.
3. <img src="https://latex.codecogs.com/gif.latex?p(y)"/></a> - Априорная вероятность, т.е. вероятность появления класса.

### Изолинии
Случайная величина x ∈ R имеет нормальное (гауссовское) распределение с параметрами µ и σ^2, если ее плотность задается выражением:  
![f](http://simfik.ru/i/f.png)  
Параметры µ и σ^2 определяют, соответственно, мат.ожидание и дисперсию нормальной случайной величины. По центральной предельной теореме среднее арифметическое независимых случайных величин с ограниченными мат.ожиданием и дисперсией стремится к нормальному распределению. Поэтому это распределение часто используется в качестве модели шума, который определяется суммой большого количества независимых друг от друга случайных факторов.  
Собственно, <a href="https://dimitridze.shinyapps.io/line/">реализация</a>  задания.

Визуализация:  
<p><img src="img\line.png" ></p>
