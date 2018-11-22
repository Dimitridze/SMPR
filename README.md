# SMPR
Language R

- [Метрические алгоритмы классификации](#Метрические-алгоритмы-классификации)
  - [1NN](#1NN)
  - [KNN](#KNN)
  - [KWNN](#KWNN)
  - [Парзеновское окно (PW)](#Парзеновское-окно-pw)
  - [Потенциальные функции (PF)](#Потенциальные-функции-pf)
  - [Алгоритм STOLP](#Алгоритм-STOLP)
 
  
  
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
  __Алгоритм__ вокруг нашей классифицируемой точки _u_ строит окружность с радиусом _h_. Далее убираем точки, которые не вошли в окружность. Затем для оставшихся, считаем _weights_, суммируем  по class, и с помощью _names(which.max(weights))_ возвращаем название доминирующего класса.
  
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

### Потенциальные функции (PF)

__Метод потенциальных функций__ - метрический классификатор, частный случай метода ближайших соседей. Позволяет с помощью простого алгоритма оценивать вес («важность») объектов обучающей выборки при решении задачи классификации.
При классификации объект проверяется на близость к объектам из обучающей выборки. Можно сказать, что объекты из обучающей выборки «заряжены» своим классом, а мера «важности» каждого из них при классификации зависит от его «заряда» и расстояния до классифицируемого объекта.

<p><img src="img\formula.png" ></p>

 Программная реализация метода потенциальных функций:
 
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
	 
Программная реализация алгоритма подбора ![gamma_i](http://latex.codecogs.com/gif.latex?%5Cgamma_%7Bi%7D):

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
В данной программе использовал прямоугольное ядро. алгоритм подбирает только силу потенциала , радиус потенциалов h я задал заранее фиксированным.

Программная реализация подбора потенциалов пока точность алгоритма, не будет меньше заданной ошибки myError
<p><img src="img\PFF2.png" ></p>

__Преимущества метода потенциальных функций:__

- Метод прост для понимания и алгоритмической реализации;
- Большое количество параметров для подбора. 

__Недостатки метода:__

- Неопределённое время работы алгоритма подбора ![gamma_i](http://latex.codecogs.com/gif.latex?%5Cgamma_%7Bi%7D);
- Параметры ![gamma_i](http://latex.codecogs.com/gif.latex?%5Cgamma_%7Bi%7D). и _h_ настраиваются слишком грубо;
- Значения параметров ![gamma_i](http://latex.codecogs.com/gif.latex?%5Cgamma_%7Bi%7D). зависят от порядка выбора объектов из выборки X.



### Алгоритм STOLP
Введём понятие *отступа*. Отступ - величина, показывающая, насколько объект является типичным представителем своего класса. Отступ равен разности между степенью близости объекта к своему классу и суммой близостей объекта ко всем остальным классам.  
![](http://latex.codecogs.com/gif.latex?M%28x_%7Bi%7D%29%3DW_%7By_%7Bi%7D%7D%28x_%7Bi%7D%2CX%5E%7Bl%7D%29-%5Cmax%20W_%7By%7D%28x_%7Bi%7D%2C%20X%5E%7Bl%7D%29)  
Все объекты обучающей выборки можно разделить на несколько типов:

1. Эталонные объекты - наиболее типичные представители своего класса
2. Неинформативные - объекты, не влияющие значительным образом на качество классификации
3. Пограничные - объекты, имеющие отступ, близкий к нулю. Незначительное изменение в выборке, алгоритме, метрике и т.п. может повлиять на их классификацию.
4. Ошибочные - объекты с отрицательными отступами, классифицируемые неверно.
5. Шумовые объекты (выбросы) - малая группа объектов с большими отрицательными отступами. Их удаление улучшает качество классификации.

Идея алгоритма STOLP состоит в том, чтобы уменьшить исходную выборку, выбрав из неё эталонные объекты. Такой подход уменьшит размер выборки, и может улучшить качество классификации.  На вход подаётся выборка, допустимый порог ошибок и порог фильтрации выбросов. Алгоритм:  

1. Удалить из выборки все выбросы (объекты, отступ которых меньше порога фильтрации выбросов).
2. Пересчитать все отступы заново, взять по одному эталону из каждого класса (объекты с наибольшим положительным отступом), и добавить их в множество эталонов.
3. Проклассифицировать объекты обучающей выборки, взяв в качестве обучающей выборки для этого множество эталонов. Посчитать число ошибок.
4. Если число ошибок меньше заданного порога, то алгоритм завершается.
5. Иначе присоединить ко множеству эталонов объект с наименьшим отступом.
6. Повторять шаги 3-5 до тех пор, пока множество эталонов и обучающая выборка не совпадут, или не сработает проверка в пункте 4.

Протестируем алгоритм на выборке ирисов Фишера.  
В качестве алгоритма классификации будем использовать kwNN(k = 10, q = 0.1)  
