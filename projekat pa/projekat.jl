using StatsModels
using GLM
using DataFrames
using CSV
using Lathe
using Plots
using Statistics
using StatsBase
using MLBase
using ROC

# 1. Priprema i provera podataka

data = DataFrame(CSV.File("boje.csv"))

dataTrain, dataTest = Lathe.preprocess.TrainTestSplit(data, .80)


# 2. Logisticka regresija

f1 = @formula(crvena ~ x + y)   # CRVENA

logisticRegressorCrvena = glm(f1, dataTrain, Binomial(), ProbitLink()) 

dataPredictedTestCrvena = predict(logisticRegressorCrvena, dataTest)

println("Predvidjeni podaci za crvenu boju: $(round.(dataPredictedTestCrvena; digits = 2))")
println()

f2 = @formula(plava ~ x + y)    # PLAVA

logisticRegressorPlava = glm(f2, dataTrain, Binomial(), ProbitLink()) 

# Testranje podataka logistickom regresijom
dataPredictedTestPlava = predict(logisticRegressorPlava, dataTest)

println("Predvidjeni podaci za plavu boju: $(round.(dataPredictedTestPlava; digits = 2))")
println()

f3 = @formula(zelena ~ x + y)   # ZELENA

logisticRegressorZelena = glm(f3, dataTrain, Binomial(), ProbitLink()) 

dataPredictedTestZelena = predict(logisticRegressorZelena, dataTest)

println("Predvidjeni podaci za zelenu boju: $(round.(dataPredictedTestZelena; digits = 2))")
println()


# 3. One-versus-one algoritam

# Prebrojavanje koliko ima podataka koji pripadaju Crvenoj klasi
brojPodatakaKojiPripadaCrvenojKlasi = 0

for i in 1 : length(dataPredictedTestCrvena)
    if dataPredictedTestCrvena[i] > 0.5
       global brojPodatakaKojiPripadaCrvenojKlasi += 1
    end
end

# Ispis broja Crvenih
println("Broj podataka koji pripadaju Crvenoj klasi je: $(brojPodatakaKojiPripadaCrvenojKlasi)")
println()

# Prebrojavanje koliko ima podataka koji pripadaju Plavojj klasi
brojPodatakaKojiPripadaPlavojKlasi = 0

for i in 1 : length(dataPredictedTestPlava)
    if dataPredictedTestPlava[i] > 0.5
       global brojPodatakaKojiPripadaPlavojKlasi += 1
    end
end

# Ispis broja Plavih
println("Broj podataka koji pripadaju Plavoj klasi je: $(brojPodatakaKojiPripadaPlavojKlasi)")
println()

# Prebrojavanje koliko ima podataka koji pripadaju Zelenojj klasi
brojPodatakaKojiPripadaZelenojKlasi = 0

for i in 1 : length(dataPredictedTestZelena)
    if dataPredictedTestZelena[i] > 0.5
       global brojPodatakaKojiPripadaZelenojKlasi += 1
    end
end

# Ispis broja Zelenih
println("Broj podataka koji pripadaju Zelenoj klasi je: $(brojPodatakaKojiPripadaZelenojKlasi)")
println()

# Provera kojoj klasi pipada najviše podataka
dataPredictedTest = [] # pomoćni niz, u koji smeštamo one podatke koje izaberemo
nazivKolone = [] # koju kolonu uzimamo iz dataTest

if brojPodatakaKojiPripadaCrvenojKlasi > brojPodatakaKojiPripadaPlavojKlasi
    if brojPodatakaKojiPripadaPlavojKlasi > brojPodatakaKojiPripadaZelenojKlasi
        global dataPredictedTest = dataPredictedTestCrvena # ubacijuemo niz predviđenih podataka za Crvenu klasu u pomoćni niz
        global nazivKolone = dataTest.crvena    # podatke iz kolone crvena smeštamo u niz nazivKolone
        println("Najvise podataka pripada Crvenoj klasi!")
    end
elseif brojPodatakaKojiPripadaPlavojKlasi > brojPodatakaKojiPripadaCrvenojKlasi
    if brojPodatakaKojiPripadaPlavojKlasi > brojPodatakaKojiPripadaZelenojKlasi
        global dataPredictedTest = dataPredictedTestPlava  # ubacijuemo niz predviđenih podataka za Plavu klasu u pomoćni niz
        global nazivKolone = dataTest.plava # podatke iz kolone plava smeštamo u niz nazivKolone
        println("Najvise podataka pripada Plavoj klasi!")
    end
elseif brojPodatakaKojiPripadaZelenojKlasi > brojPodatakaKojiPripadaCrvenojKlasi
    if brojPodatakaKojiPripadaZelenojKlasi > brojPodatakaKojiPripadaPlavojKlasi
        global dataPredictedTest = dataPredictedTestZelena # ubacijuemo niz predviđenih podataka za Yelenu klasu u pomoćni niz
        global nazivKolone = dataTest.zelena    # podatke iz kolone zelena smeštamo u niz nazivKolone
        println("Najvise podataka pripada Zelenoj klasi!")
    end
end
println()



# 4. Analiza kvaliteta modela

# kreiranje matrice
dataPredictedTestClass = repeat(0:0, length(dataPredictedTest))

for i in 1:length(dataPredictedTest)
    if (dataPredictedTest[i] > 0.5)
        dataPredictedTestClass[i] = 1
    end
end

FPTest = 0 # false positives
FNTest = 0 # false negatives
TPTest = 0 # true positives
TNTest = 0 # true negatives

for i in 1:length(dataPredictedTestClass)
    if nazivKolone[i] == 0 && dataPredictedTestClass[i] == 0
        global TNTest += 1
    elseif nazivKolone[i] == 0 && dataPredictedTestClass[i] == 1
        global FPTest += 1
    elseif nazivKolone[i] == 1 && dataPredictedTestClass[i] == 0
        global FNTest += 1
    elseif nazivKolone[i] == 1 && dataPredictedTestClass[i] == 1
        global TPTest += 1
    end
end


# Ocena kvaliteta klasifikacije

accuracyTest = (TPTest + TNTest) / (TPTest + TNTest + FPTest + FNTest)

sensitivityTest = TPTest / (TPTest + FNTest)

specificityTest = TNTest / (TNTest + FPTest)


println("TP = $TPTest, FP = $FPTest, TN = $TNTest, FN = $FNTest\n")

println("Preciznost :$accuracyTest\n")

println("Osetljivost :$sensitivityTest\n")

println("Specificnost :$specificityTest\n") 

# Roc kriva
rocTest = ROC.roc(dataPredictedTest, nazivKolone, true)

aucTest = AUC(rocTest) 
println("Povrsina ispod krive u procentima je: $aucTest\n")

if (aucTest > 0.9)
    println("Klasifikator je jako dobar")
elseif (aucTest > 0.8)
    println("Klasifikator je veoma dobar")
elseif (aucTest > 0.7)
    println("Klasifikator je dosta dobar")
elseif (aucTest > 0.5)
    println("Klasifikator je relativno dobar")
else
    println("Klasifikator je los")
end

# crtanje ROC krive
plot = scatter(rocTest, label = "ROC curve", legend = :bottomright)
savefig(plot, "ROC-curve.html")