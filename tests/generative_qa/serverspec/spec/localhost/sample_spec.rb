require 'spec_helper'
require 'serverspec_extended_types'

describe os[:family] do
  it { should match /ubuntu|windows|darwin/}
end

describe os[:release], :if => os[:family] == 'ubuntu' do
  it { should be >= "16.04" }
end

describe os[:release], :if => os[:family] == 'windows' do
  it { should be >= '7' }
end


describe virtualenv("/home/admin/envbpr") do
  it { should be_virtualenv }
end


describe virtualenv("/home/admin/envbpr") do
  its(:pip_version) { should match '22.0.4' }
end


describe virtualenv("/home/admin/envbpr") do
  its(:python_version) { should match /^3\.7\./ }
end

describe virtualenv("/home/admin/envbpr") do
  its(:pip_freeze) { should include('requests' => '2.22.0') }
  its(:pip_freeze) { should include('pytest' => '7.1.2') }
  its(:pip_freeze) { should include('bpr @ file:///data/home/admin/Projects/bpr') }
  its(:pip_freeze) { should include('transformers' => '4.6.0') }
  its(:pip_freeze) { should include('pyTelegramBotAPI' => '3.6.7') }
  its(:pip_freeze) { should include('numpy' => '1.18.0') }
  its(:pip_freeze) { should include('scikit-learn' => '0.21.2') }
  its(:pip_freeze) { should include('matplotlib-inline' => '0.1.3') }
  its(:pip_freeze) { should include('tqdm' => '4.62.0') }
  its(:pip_freeze) { should include('sacrebleu' => '2.0.0') }
  its(:pip_freeze) { should include('tokenizers' => '0.10.3') }
  its(:pip_freeze) { should include('torch' => '1.6.0') } 
end

describe file('/home/admin/Projects/DeepPavlov-1') do
  it {should exist}
end