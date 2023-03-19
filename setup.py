from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="ktml",
    version="0.3",
    description="""Final Project TM10007ML""",
    license="Apache 2.0 License",
    author="Martijn Starmans, Karin van Garderen, Hakim Achterberg",
    author_email="m.starmans@erasmusmc.nl",
    install_requires=required,
    include_package_data=True,
    packages=[
        "brats",
        "hn",
        "adni",
        "worclipo",
        "worcliver",
        "worccrlm",
        "worcdesmoid",
        "worcgist"
    ],
)
