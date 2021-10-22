clear all
close all
clc

rand('seed',30)


% 1 - Diabetes
% 2 - Alcohol
% 3 - Life expectancy
% 4 - Rural Population
% 5 - GDP/capita
% 6 - Smoking
% 7 - Adolescent Fertility Rate
% 8 - Obesity Rate
% 9 - GHG emission
% 10 - Population


%% Hyperparameters

m=3; % No of clusters
maxit=200; % No of iteration


%% Loading the data

% COVID-19 Mortality cases
TC=readtable('CovidData.xlsx'); 

% Explanatory variables
TV=readtable('Variables.xlsx');
TV(:,6)=table(table2array(TV(:,6))./table2array(TV(:,12))); % Rural population is applied as the percentile of population. In these database, the total number of rural population were given.

% Name of explanatory variables for plot legends
VNames=readtable('VariableNames.xlsx');
VNames=string(table2array(VNames));


%% Listwise deletion of missing data

% Obtaining the countries, which are in both Covid and Variables database
index=ismember(table2array(TV(:,1)),table2array(TC(:,1))); 
TV(~index,:)=[];
index=ismember(table2array(TC(:,1)),table2array(TV(:,1)));
TC(~index,:)=[];

% Sorting the data according to countries name based on the alphabet (Now
% the index of the rows belong to the same country)
TC=sortrows(TC,1);
TV=sortrows(TV,1);

% find NaN-s in the Variables database and listwise delete them
index=sum(~isnan(table2array(TV(:,3:end))),2)==size(table2array(TV(:,3:end)),2); 
TC(~index,:)=[];
TV(~index,:)=[];

% find NaN-s in the COVID-19 database and listwise delete them
index=sum(~isnan(table2array(TC(:,5))),2)==size(table2array(TC(:,5)),2);
TC(~index,:)=[];
TV(~index,:)=[];

% Obtaining the variables and COVID Data in a Matrix form
Variables=table2array(TV(:,3:end-1));
DeathCase=table2array(TC(:,5));


%% Inicialization of the algorithm

N=size(Variables,1); % Number of data
NoComp=size(Variables,2); % Number of explanatory variable
p=ones(1,m)*1/m; % Initial cluster proportions

thetae=ones(1,m)*100; % Initial Weibull parameters
betae=ones(1,m);

% Initial multivariate Gaussian parameters
gm = fitgmdist(Variables,m,'Options',statset('Display','final','MaxIter',300),'RegularizationValue',0.1);
v=gm.mu;
P=gm.Sigma;
for j=1:m
    P(:,:,j)=diag(diag(P(:,:,j))); % We only needed the diagonal elements for interpretable results.
end

U=initfc(m,N); % Initial membership values
U=U';

% Iteration of the algorithm
for it=1:maxit
    % E-step
    
    for j=1:m
        pzj(:,j) = wblpdf(DeathCase,thetae(j),betae(j)); % Calculating Equation 8
        pxj(:,j) = mvnpdf(Variables,v(j,:),P(:,:,j)); % Calculating Equation 9
    end  
    dum=pzj.*pxj.*repmat(p,N,1); % Numerator of Equation 11
    pjtx=dum./repmat(sum(dum,2),1,m); % Calculating Equation 11
    
    % M-step
    
    % Unconditional cluster probability
    p=sum(pjtx)/N; % Calculating Equation 12
    
    for j=1:m
        %cluster centre
        v(j,:)=sum(Variables.*repmat(pjtx(:,j),1,NoComp))/(N*p(j)); % Calculating Equation 13
        
        %Covariance matrix
        dum=(Variables-repmat(v(j,:),N,1)).*sqrt(repmat(pjtx(:,j),1,NoComp)); % Calculating Equation 14
        P(:,:,j)=dum'*dum/(N*p(j)); % Calculating Equation 14
        P(:,:,j)=diag(diag(P(:,:,j))); % We only needed the diagonal elements for interpretable results.
                
        %Local Weibull model
        dum= wblfit(DeathCase,[],[],pjtx(:,j)); % Minimizing Equation 7
        thetae(j)=dum(1); betae(j)=dum(2);
    end
end


%% Plot the Weibull distribution function


x=0:0.1:500; % Domain of Weibull distribution Function

colorset=['g' 'b' 'r--'  'k' 'm' 'k'];
Markset={'-';'--';'-.'};


figure(1086)

subplot(5,2,1)
hold on
for j=1:m
    y = wblcdf(x,thetae(j),betae(j));
    plot(x,1-y,'LineStyle',Markset{j,1},'Color',colorset(j),'LineWidth',2)   
end
xlabel('Death cases per 100K population')
ylabel('Probability')
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5')
%saveas(gcf, 'survival_rul10.eps','epsc')



%% Plot the membership functions

% % Inverse of covariance matrix is needed
lambda=1;
for j=1:m
    F(:,:,j)=P(:,:,j);
    iF=lambda*inv(F(:,:,j))+(1-lambda)*eye(size(F(:,:,j),1));
    M(:,:,j)=iF;
end

colorset=['g' 'b' 'r'  'k' 'm' 'k'];
Markset={'-';'--';'-.'};
%Plot the Mem functions


for i=1:size(Variables,2)
    xd = linspace(min(Variables(:,i)),max(Variables(:,i)),200);
    
    subplot(5,2,i+1)
    hold on
    for j=1:m
        iM=M(:,:,j);
        y1 = exp(-1/2*(xd-v(j,i)).^2*iM(i,i));
        xd1=xd;
        plot(xd1,y1,'LineStyle',Markset{j,1},'Color',colorset(j),'LineWidth',2)
        axis([min(xd1)  max(xd1) 0 1]);
    end
    xlabel(VNames{i})
    ylabel('Membership')
    hold off

end



%% Countries in Clusters

[~,dum]=max(pjtx,[],2);
CountryCluster=[string(table2array(TV(:,1:2))) string(dum)];

% Histogram of data for clusters

Cluster=double(CountryCluster(:,3));
%clf
figure(1090)
subplot(5,2,1)
hold on

Nbin= 20;
dum=((max(DeathCase) - min(DeathCase))/Nbin) ;

h1=DeathCase(Cluster==1,:);
h2=DeathCase(Cluster==2,:);
h3=DeathCase(Cluster==3,:);

h1 = histogram(h1,Nbin);
h2 = histogram(h2,Nbin);
h3 = histogram(h3,Nbin);

h1.BinWidth = dum;
h1.FaceColor = 'green';

h2.BinWidth = dum;
h2.FaceColor = 'blue';

h3.BinWidth = dum;
h3.FaceColor = 'red';

xlabel('Death cases per 100K population')
ylabel('Number of countries')
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5')
xlim([0 620])



for i=1:size(Variables,2)

subplot(5,2,i+1)
hold on


dum=((max(Variables(:,i)) - min(Variables(:,i)))/Nbin) ;

var=i;

h1=Variables(Cluster==1,var);
h2=Variables(Cluster==2,var);
h3=Variables(Cluster==3,var);

h1 = histogram(h1,Nbin);
h2 = histogram(h2,Nbin);
h3 = histogram(h3,Nbin);


h1.BinWidth = dum;
h1.FaceColor = 'green';

h2.BinWidth = dum;
h2.FaceColor = 'blue';

h3.BinWidth = dum;
h3.FaceColor = 'red';
xlabel(VNames{i})
ylabel('Number of Countries')

end

%saveas(gcf,'histogram2.eps','epsc')


%% Disp Controll Values

disp('Covariance Matrix of Clusters')
P
disp('Proportion of the Clusters')
p
disp('Scale Parameter of Weibull Distribution')
thetae
disp('Shape Parameter of Weibull Distribution')
betae
disp('Center of Clusters')
v





