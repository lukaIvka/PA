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
data = DataFrame(CSV.File("figure.csv"))

# Podela na skup za obuku i testiranja
dataTrain, dataTest = Lathe.preprocess.TrainTestSplit(data, .80)


# 2. Logisticka regresija

# Formiranje formule za logistički regresor
f1 = @formula(pravougaonik ~ x + y)   # pravougaonik

# Poziv logistički regresora
logisticRegressorPravougaonik = glm(f1, dataTrain, Binomial(), ProbitLink()) 

# Testranje podataka logistickom regresijom
dataPredictedTestPravougaonik = predict(logisticRegressorPravougaonik, dataTest)

# Ispis predviđenih podataka
println("Predvidjeni podaci za pravougaonik: $(round.(dataPredictedTestPravougaonik; digits = 2))\n")

# Formiranje formule za logistički regresor
f2 = @formula(kvadrat ~ x + y)    # kvadrat

# Poziv logistički regresora
logisticRegressorKvadat = glm(f2, dataTrain, Binomial(), ProbitLink()) 

# Testranje podataka logistickom regresijom
dataPredictedTestKvadrat = predict(logisticRegressorKvadat, dataTest)

# Ispis predviđenih podataka
println("Predvidjeni podaci za kvadrat: $(round.(dataPredictedTestKvadrat; digits = 2))\n")

# Formiranje formule za logistički regresor
f3 = @formula(trougao ~ x + y)   # trougao

# Poziv logistički regresora
logisticRegressorTrougao = glm(f3, dataTrain, Binomial(), ProbitLink()) 

# Testranje podataka logistickom regresijom
dataPredictedTestTrougao = predict(logisticRegressorTrougao, dataTest)

# Ispis predviđenih podataka
println("Predvidjeni podaci za trougao: $(round.(dataPredictedTestTrougao; digits = 2))\n")


f4 = @formula(krug ~ x + y)
# Poziv logistički regresora
logisticRegressorKrug = glm(f4, dataTrain, Binomial(), ProbitLink()) 
# Testranje podataka logistickom regresijom
dataPredictedTestKrug = predict(logisticRegressorKrug, dataTest)
# Ispis predviđenih podataka
println("Predvidjeni podaci za krug: $(round.(dataPredictedTestKrug; digits = 2))\n")


# 3. One-versus-one algoritam

# Provera koji podatak pripada kojoj klasi

# Proveravamo koji podaci pripadaju klasi Pravougaonik
matricaPravougaonik = repeat(0:0, length(dataPredictedTestPravougaonik))

# formiranje matrice
for i in 1:length(dataPredictedTestPravougaonik)
    if (dataPredictedTestPravougaonik[i] > 0.5)
        matricaPravougaonik[i] = 1
    end
end

indexPravougaonik = [] # pomoćni niz za indexe podataka koji pripadaju klasi Pravougaonik

# punjenje pomoćnog niza sa indexima
for i in 1 : length(matricaPravougaonik)
    if matricaPravougaonik[i] == 1
        append!(indexPravougaonik, i)
    end
end

println("$indexPravougaonik\n") # ispis indexa podataka koji pripadaju klasi Pravougaonik

# Proveravamo koji podaci pripadaju klasi Kvadrat
matricaKvadrat = repeat(0:0, length(dataPredictedTestKvadrat))

for i in 1:length(dataPredictedTestKvadrat)
    if (dataPredictedTestKvadrat[i] > 0.5)
        matricaKvadrat[i] = 1
    end
end

indexKvadrat = [] # pomoćni niz za indexe podataka koji pripadaju Plavoj klasi

for i in 1 : length(matricaKvadrat)
    if matricaKvadrat[i] == 1
        append!(indexKvadrat, i)
    end
end

println("$indexKvadrat\n") # ispis indexa podataka koji pripadaju klasi Kvadrat

# Proveravamo koji podaci pripadaju klasi Trougao
matricaTrougao = repeat(0:0, length(dataPredictedTestTrougao))

for i in 1:length(dataPredictedTestTrougao)
    if (dataPredictedTestTrougao[i] > 0.5)
        matricaTrougao[i] = 1
    end
end

indexTrougao = [] # pomoćni niz za indexe podataka koji pripadaju klasi Trougao

for i in 1 : length(matricaTrougao)
    if matricaTrougao[i] == 1
        append!(indexTrougao, i)
    end
end

println("$indexTrougao\n") # ispis indexa podataka koji pripadaju klasi Trougao




# Proveravamo koji podaci pripadaju klasi Krug
matricaKrug = repeat(0:0, length(dataPredictedTestKrug))
# formiranje matrice
for i in 1:length(dataPredictedTestKrug)
    if (dataPredictedTestKrug[i] > 0.5)
        matricaKrug[i] = 1
    end
end
indexKrug = [] # pomoćni niz za indexe podataka koji pripadaju klasi Krug
# punjenje pomoćnog niza sa indexima
for i in 1 : length(matricaKrug)
    if matricaKrug[i] == 1
        append!(indexKrug, i)
    end
end







# Prebrojavanje koliko ima podataka koji pripadaju klasi Pravougaonik
brojPodatakaKojiPripadaKlasiPravougaonik = 0

for i in 1 : length(dataPredictedTestPravougaonik)
    if dataPredictedTestPravougaonik[i] > 0.5
       global brojPodatakaKojiPripadaKlasiPravougaonik += 1
    end
end

# Ispis broja Pravougaonika
println("Broj podataka koji pripadaju klasi Pravougaonik je: $(brojPodatakaKojiPripadaKlasiPravougaonik)\n")

# Prebrojavanje koliko ima podataka koji pripadaju klasi Kvadrat
brojPodatakaKojiPripadaKlasiKvadrat = 0

for i in 1 : length(dataPredictedTestKvadrat)
    if dataPredictedTestKvadrat[i] > 0.5
       global brojPodatakaKojiPripadaKlasiKvadrat += 1
    end
end

# Ispis broja Kvadrata
println("Broj podataka koji pripadaju klasi Kvadrat je: $(brojPodatakaKojiPripadaKlasiKvadrat)\n")

# Prebrojavanje koliko ima podataka koji pripadaju klasi Trougao
brojPodatakaKojiPripadaKlasiTrougao = 0

for i in 1 : length(dataPredictedTestTrougao)
    if dataPredictedTestTrougao[i] > 0.5
       global brojPodatakaKojiPripadaKlasiTrougao += 1
    end
end

# Ispis broja Trouglova
println("Broj podataka koji pripadaju klasi Trougao je: $(brojPodatakaKojiPripadaKlasiTrougao)\n")




# Prebrojavanje koliko ima podataka koji pripadaju klasi Krug
brojPodatakaKojiPripadaKlasiKrug = 0

for i in 1 : length(dataPredictedTestKrug)
    if dataPredictedTestKrug[i] > 0.5
       global brojPodatakaKojiPripadaKlasiKrug += 1
    end
end
# Ispis broja Krugova
println("Broj podataka koji pripadaju klasi Krug je: $(brojPodatakaKojiPripadaKlasiKrug)\n")





# Provera kojoj klasi pipada najviše podataka
dataPredictedTest = [] # pomoćni niz, u koji smeštamo one podatke koje izaberemo
nazivKolone = [] # koju kolonu uzimamo iz dataTest

if brojPodatakaKojiPripadaKlasiPravougaonik > brojPodatakaKojiPripadaKlasiKvadrat
    if brojPodatakaKojiPripadaKlasiPravougaonik > brojPodatakaKojiPripadaKlasiTrougao
        if brojPodatakaKojiPripadaKlasiPravougaonik > brojPodatakaKojiPripadaKlasiKrug
        global dataPredictedTest = dataPredictedTestPravougaonik # ubacijuemo niz predviđenih podataka za klasu Pravougaonik u pomoćni niz
        global nazivKolone = dataTest.pravougaonik    # podatke iz kolone pravougaonik smeštamo u niz nazivKolone
        println("Najvise podataka pripada klasi Pravougaonik, pa nju uzimamo!\n")
        end
    end
elseif brojPodatakaKojiPripadaKlasiKvadrat > brojPodatakaKojiPripadaKlasiPravougaonik
    if brojPodatakaKojiPripadaKlasiKvadrat > brojPodatakaKojiPripadaKlasiTrougao
        if brojPodatakaKojiPripadaKlasiKvadrat > brojPodatakaKojiPripadaKlasiKrug
        global dataPredictedTest = dataPredictedTestKvadrat  # ubacijuemo niz predviđenih podataka za klasu Kvadrat u pomoćni niz
        global nazivKolone = dataTest.kvadrat # podatke iz kolone kvadrat smeštamo u niz nazivKolone
        println("Najvise podataka pripada klasi Kvadrat, pa nju uzimamo!\n")
         end
    end
elseif brojPodatakaKojiPripadaKlasiTrougao > brojPodatakaKojiPripadaKlasiPravougaonik
    if brojPodatakaKojiPripadaKlasiTrougao > brojPodatakaKojiPripadaKlasiKvadrat
        if brojPodatakaKojiPripadaKlasiTrougao > brojPodatakaKojiPripadaKlasiKrug
        global dataPredictedTest = dataPredictedTestTrougao # ubacijuemo niz predviđenih podataka za klasu Trougao u pomoćni niz
        global nazivKolone = dataTest.trougao    # podatke iz kolone trougao smeštamo u niz nazivKolone
        println("Najvise podataka pripada klasi Trougao, pa nju uzimamo!\n")
        end
    end
elseif brojPodatakaKojiPripadaKlasiKrug > brojPodatakaKojiPripadaKlasiPravougaonik
    if brojPodatakaKojiPripadaKlasiKrug > brojPodatakaKojiPripadaKlasiKvadrat
        if brojPodatakaKojiPripadaKlasiKrug > brojPodatakaKojiPripadaKlasiTrougao
        global dataPredictedTest = dataPredictedTestKrug # ubacijuemo niz predviđenih podataka za klasu Krug u pomoćni niz
        global nazivKolone = dataTest.krug    # podatke iz kolone trougao smeštamo u niz nazivKolone
        println("Najvise podataka pripada klasi Krug, pa nju uzimamo!\n")
         end
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