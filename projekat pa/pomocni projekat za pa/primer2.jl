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

# Ucitavanje podataka
data = DataFrame(CSV.File("boje.csv"))

# Podela na skup za obuku i testiranja
dataTrain, dataTest = Lathe.preprocess.TrainTestSplit(data, .80)


# 2. Logisticka regresija

# Formiranje formule za logistički regresor
f1 = @formula(crvena ~ x + y)   # CRVENA

# Poziv logistički regresora
logisticRegressorCrvena = glm(f1, dataTrain, Binomial(), ProbitLink()) 

# Testranje podataka logistickom regresijom
dataPredictedTestCrvena = predict(logisticRegressorCrvena, dataTest)

# Ispis predviđenih podataka
println("Predvidjeni podaci za crvenu boju: $(round.(dataPredictedTestCrvena; digits = 2))\n")

# Formiranje formule za logistički regresor
f2 = @formula(plava ~ x + y)    # PLAVA

# Poziv logistički regresora
logisticRegressorPlava = glm(f2, dataTrain, Binomial(), ProbitLink()) 

# Testranje podataka logistickom regresijom
dataPredictedTestPlava = predict(logisticRegressorPlava, dataTest)

# Ispis predviđenih podataka
println("Predvidjeni podaci za plavu boju: $(round.(dataPredictedTestPlava; digits = 2))\n")

# Formiranje formule za logistički regresor
f3 = @formula(zelena ~ x + y)   # ZELENA

# Poziv logistički regresora
logisticRegressorZelena = glm(f3, dataTrain, Binomial(), ProbitLink()) 

# Testranje podataka logistickom regresijom
dataPredictedTestZelena = predict(logisticRegressorZelena, dataTest)

# Ispis predviđenih podataka
println("Predvidjeni podaci za zelenu boju: $(round.(dataPredictedTestZelena; digits = 2))\n")


# 3. One-versus-one algoritam

# Provera koji podatak pripada kojoj klasi

# Proveravamo koji podaci pripadaju Crvenoj klasi
matricaCrvene = repeat(0:0, length(dataPredictedTestCrvena))

# formiranje matrice
for i in 1:length(dataPredictedTestCrvena)
    if (dataPredictedTestCrvena[i] > 0.5)
        matricaCrvene[i] = 1
    end
end

indexCrvene = [] # pomoćni niz za indexe podataka koji pripadaju Crvenoj klasi

# punjenje pomoćnog niza sa indexima
for i in 1 : length(matricaCrvene)
    if matricaCrvene[i] == 1
        append!(indexCrvene, i)
    end
end

println("$indexCrvene\n") # ispis indexa podataka koji pripadaju Crvenoj klasi

# Proveravamo koji podaci pripadaju Plavoj klasi
matricaPlava = repeat(0:0, length(dataPredictedTestPlava))

for i in 1:length(dataPredictedTestCrvena)
    if (dataPredictedTestPlava[i] > 0.5)
        matricaPlava[i] = 1
    end
end

indexPlava = [] # pomoćni niz za indexe podataka koji pripadaju Plavoj klasi

for i in 1 : length(matricaPlava)
    if matricaPlava[i] == 1
        append!(indexPlava, i)
    end
end

println("$indexPlava\n") # ispis indexa podataka koji pripadaju Plavoj klasi

# Proveravamo koji podaci pripadaju Zelenoj klasi
matricaZelena = repeat(0:0, length(dataPredictedTestZelena))

for i in 1:length(dataPredictedTestZelena)
    if (dataPredictedTestZelena[i] > 0.5)
        matricaZelena[i] = 1
    end
end

indexZelena = [] # pomoćni niz za indexe podataka koji pripadaju Zelenoj klasi

for i in 1 : length(matricaZelena)
    if matricaZelena[i] == 1
        append!(indexZelena, i)
    end
end

println("$indexZelena\n") # ispis indexa podataka koji pripadaju Zelenoj klasi

# Prebrojavanje koliko ima podataka koji pripadaju Crvenoj klasi
brojPodatakaKojiPripadaCrvenojKlasi = 0

for i in 1 : length(dataPredictedTestCrvena)
    if dataPredictedTestCrvena[i] > 0.5
       global brojPodatakaKojiPripadaCrvenojKlasi += 1
    end
end

# Ispis broja Crvenih
println("Broj podataka koji pripadaju Crvenoj klasi je: $(brojPodatakaKojiPripadaCrvenojKlasi)\n")

# Prebrojavanje koliko ima podataka koji pripadaju Plavojj klasi
brojPodatakaKojiPripadaPlavojKlasi = 0

for i in 1 : length(dataPredictedTestPlava)
    if dataPredictedTestPlava[i] > 0.5
       global brojPodatakaKojiPripadaPlavojKlasi += 1
    end
end

# Ispis broja Plavih
println("Broj podataka koji pripadaju Plavoj klasi je: $(brojPodatakaKojiPripadaPlavojKlasi)\n")

# Prebrojavanje koliko ima podataka koji pripadaju Zelenojj klasi
brojPodatakaKojiPripadaZelenojKlasi = 0

for i in 1 : length(dataPredictedTestZelena)
    if dataPredictedTestZelena[i] > 0.5
       global brojPodatakaKojiPripadaZelenojKlasi += 1
    end
end

# Ispis broja Zelenih
println("Broj podataka koji pripadaju Zelenoj klasi je: $(brojPodatakaKojiPripadaZelenojKlasi)\n")

# Provera kojoj klasi pipada najviše podataka
dataPredictedTest = [] # pomoćni niz, u koji smeštamo one podatke koje izaberemo
nazivKolone = [] # koju kolonu uzimamo iz dataTest

if brojPodatakaKojiPripadaCrvenojKlasi > brojPodatakaKojiPripadaPlavojKlasi
    if brojPodatakaKojiPripadaPlavojKlasi > brojPodatakaKojiPripadaZelenojKlasi
        global dataPredictedTest = dataPredictedTestCrvena # ubacijuemo niz predviđenih podataka za Crvenu klasu u pomoćni niz
        global nazivKolone = dataTest.crvena    # podatke iz kolone crvena smeštamo u niz nazivKolone
        println("Najvise podataka pripada Crvenoj klasi, pa nju uzimamo!\n")
    end
elseif brojPodatakaKojiPripadaPlavojKlasi > brojPodatakaKojiPripadaCrvenojKlasi
    if brojPodatakaKojiPripadaPlavojKlasi > brojPodatakaKojiPripadaZelenojKlasi
        global dataPredictedTest = dataPredictedTestPlava  # ubacijuemo niz predviđenih podataka za Plavu klasu u pomoćni niz
        global nazivKolone = dataTest.plava # podatke iz kolone plava smeštamo u niz nazivKolone
        println("Najvise podataka pripada Plavoj klasi, pa nju uzimamo!\n")
    end
elseif brojPodatakaKojiPripadaZelenojKlasi > brojPodatakaKojiPripadaCrvenojKlasi
    if brojPodatakaKojiPripadaZelenojKlasi > brojPodatakaKojiPripadaPlavojKlasi
        global dataPredictedTest = dataPredictedTestZelena # ubacijuemo niz predviđenih podataka za Yelenu klasu u pomoćni niz
        global nazivKolone = dataTest.zelena    # podatke iz kolone zelena smeštamo u niz nazivKolone
        println("Najvise podataka pripada Zelenoj klasi, pa nju uzimamo!\n")
    end
end

println("Ispis klase koja ima najvise pozitivnih klasifikacija: $(round.(dataPredictedTest; digits = 2))\n")


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

# dodela vrednosti za FPTest, FNTest, TPTest i TNTest
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

# accuracy (preciznost) = (TP+TN)/(TP+TN+FP+FN) = (TP+TN)/(P+N)
accuracyTest = (TPTest + TNTest) / (TPTest + TNTest + FPTest + FNTest)

# sensitivity (osetljivost, True positive rates) = TP/(TP+FN) = TP/P
sensitivityTest = TPTest / (TPTest + FNTest)

# specificity (specifičnost, True negative rates) = TN/(TN+FP) = TN/N
specificityTest = TNTest / (TNTest + FPTest)


println("TP = $TPTest, FP = $FPTest, TN = $TNTest, FN = $FNTest\n")

println("Preciznost za test skup je $accuracyTest\n")

println("Osetljivost za test skup je $sensitivityTest\n")

println("Specificnost za test skup je $specificityTest\n") 

# Roc kriva
rocTest = ROC.roc(dataPredictedTest, nazivKolone, true)

aucTest = AUC(rocTest) # predstavlja objektivnu meru kvaliteta klasifikatora
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